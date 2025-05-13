# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook Purpose
# MAGIC
# MAGIC * Ingestion Excel Document with Autoloader and Spark Structured Streaming
# MAGIC   * Using OSS [Unstructured API](https://docs.unstructured.io/open-source/introduction/overview) for Ingestion Excel files
# MAGIC * Convert content into Markdown text
# MAGIC * Save to a table

# COMMAND ----------

# MAGIC %pip install -U -qqq markdownify==0.12.1 "unstructured[local-inference, all-docs]==0.14.4" unstructured-client==0.22.0 nltk==3.8.1 "pdfminer.six==20221105"
# MAGIC %pip install databricks-sdk -U -q
# MAGIC %pip install tiktoken -q

# COMMAND ----------

# MAGIC %run ./00-helpers

# COMMAND ----------

install_apt_get_packages(["poppler-utils", "tesseract-ocr"])

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC # Config

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_data")
dbutils.widgets.text("volume", "excel")
dbutils.widgets.text("checkpoints_vol", "checkpoints")
dbutils.widgets.text("table_prefix", "excel")
dbutils.widgets.dropdown("reset_data", "true", ["true", "false"])

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
checkpoint_vol = dbutils.widgets.get("checkpoints_vol")
table_prefix = dbutils.widgets.get("table_prefix")
reset_data = dbutils.widgets.get("reset_data") == "true"

print(f"Use Unit Catalog: {catalog}")
print(f"Use Schema: {schema}")
print(f"Use Volume: {volume}")
print(f"Use Checkpoint Volume: {checkpoint_vol}")

# COMMAND ----------

pipeline_config = {
    "file_format": "xlsx",
    "source_path": f"/Volumes/{catalog}/{schema}/{volume}",
    "checkpoint_path": f"/Volumes/{catalog}/{schema}/{checkpoint_vol}",
    "raw_files_table_name": f"{catalog}.{schema}.{table_prefix}_raw_files",
    "parsed_file_table_name": f"{catalog}.{schema}.{table_prefix}_parsed_text",
    "prepared_text_table_name": f"{catalog}.{schema}.{table_prefix}_enriched_parsed_text"
}

# COMMAND ----------

if reset_data:
    print("Delete checkpoints volume folders ...")
    dbutils.fs.rm(f"/Volumes/{catalog}/{schema}/{checkpoint_vol}/{pipeline_config['raw_files_table_name'].split('.')[-1]}", recurse=True)
    dbutils.fs.rm(f"/Volumes/{catalog}/{schema}/{checkpoint_vol}/{pipeline_config['parsed_file_table_name'].split('.')[-1]}", recurse=True)
    dbutils.fs.rm(f"/Volumes/{catalog}/{schema}/{checkpoint_vol}/{pipeline_config['prepared_text_table_name'].split('.')[-1]}", recurse=True)

    print("Delete tables ...")
    spark.sql(f"DROP TABLE IF EXISTS {pipeline_config['raw_files_table_name']}")
    spark.sql(f"DROP TABLE IF EXISTS {pipeline_config['parsed_file_table_name']}")
    spark.sql(f"DROP TABLE IF EXISTS {pipeline_config['prepared_text_table_name']}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Ingestion and Processing

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze: Ingsetion excel file as binary using Autoloader

# COMMAND ----------

df_raw_bronze = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("pathGlobfilter", f"*.{pipeline_config.get('file_format')}")
    .load(pipeline_config.get('source_path'))
)

# COMMAND ----------

from pyspark.sql.functions import split, element_at

df_raw_bronze \
    .withColumn("file_type", element_at(split('path', '\\.'), -1)) \
    .writeStream.trigger(availableNow=True).option(
        "checkpointLocation",
        f"{pipeline_config.get('checkpoint_path')}/{pipeline_config.get('raw_files_table_name').split('.')[-1]}",
).toTable(pipeline_config.get("raw_files_table_name")).awaitTermination()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver: Parse PDF using Unstructured OSS API and convert content to markdown 
# MAGIC
# MAGIC - wrap the parsing in a Pandas UDF
# MAGIC - Unstructured would parse each tab of an excel as a table
# MAGIC - Convert table to markdonw syntax because LLM understand markdown syntax well
# MAGIC - Depending on the models sometime xml syntax is better for some LLMs but in general markdown works well across LLMs

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, col
import pandas as pd

PARSED_IMG_DIR = f"/Volumes/{catalog}/{schema}/{volume}/parsed_images"

@pandas_udf("string")
def process_pdf_bytes(contents: pd.Series) -> pd.Series:
    from unstructured.partition.xlsx import partition_xlsx
    import pandas as pd
    import re
    import io
    from markdownify import markdownify as md

    def perform_partition(raw_doc_contents_bytes):
        xlsx = io.BytesIO(raw_doc_contents_bytes)
        raw_elements = partition_xlsx(
            file=xlsx,
            infer_table_structure=True,
            lenguages=["eng"],
            strategy="hi_res",                                   
            extract_image_block_types=["Table"],
            extract_image_block_output_dir=PARSED_IMG_DIR,
        )

        text_content = ""
        for section in raw_elements:
            # Tables are parsed seperatly, add a \n to give the chunker a hint to split well.
            if section.category == "Table":
                if section.metadata is not None:
                    if section.metadata.text_as_html is not None:
                        # convert table to markdown
                        text_content += "\n" + md(section.metadata.text_as_html) + "\n"
                    else:
                        text_content += " " + section.text
                else:
                    text_content += " " + section.text
            # Other content often has too-aggresive splitting, merge the content
            else:
                text_content += " " + section.text        

        return text_content
    return pd.Series([perform_partition(content) for content in contents])

# COMMAND ----------

df_parsed_silver = (
    spark.readStream.table(pipeline_config.get("raw_files_table_name"))
        .withColumn("text", process_pdf_bytes("content"))
)

# COMMAND ----------

silver_success_query = (
    df_parsed_silver.drop("content").writeStream.trigger(availableNow=True)
    .option(
        "checkpointLocation",
        f"{pipeline_config.get('checkpoint_path')}/{pipeline_config.get('parsed_file_table_name').split('.')[-1]}",
    ).toTable(pipeline_config.get("parsed_file_table_name"))
).awaitTermination()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold: Enrich with token count
# MAGIC
# MAGIC - Add token count metadata

# COMMAND ----------

from pyspark.sql.functions import udf
import tiktoken

@udf("int")
def count_tokens(txt):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(txt, disallowed_special=())
    return len(tokens)

# COMMAND ----------

df_enriched_gold = (
    spark.readStream.table(pipeline_config.get("parsed_file_table_name"))
    .withColumn("token_count", count_tokens(col("text")))
)

gold_query = (
    df_enriched_gold.writeStream.trigger(availableNow=True)
    .option(
        "checkpointLocation",
        f"{pipeline_config.get('checkpoint_path')}/{pipeline_config.get('prepared_text_table_name').split('.')[-1]}",
    ).toTable(pipeline_config.get("prepared_text_table_name"))
).awaitTermination()