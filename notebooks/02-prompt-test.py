# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook Purpose
# MAGIC
# MAGIC Exam the parsed tables and perform prompt test on them

# COMMAND ----------

# MAGIC %pip install databricks-sdk -U -q
# MAGIC %pip install openai -U -q

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parsed from PDF files

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG fins_genai;
# MAGIC USE SCHEMA unstructured_data;

# COMMAND ----------

df_pdf = spark.table("excel_pdf_enriched_parsed_text")
display(df_pdf)

# COMMAND ----------

from IPython.display import Markdown as md

md_content_pdf = df_pdf.select("text").collect()[0].text
md(md_content_pdf)

# COMMAND ----------

md_content_pdf1 = df_pdf.select("text").collect()[1].text
md(md_content_pdf1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## parsed from EXCEL file

# COMMAND ----------

df_excel = spark.table("excel_enriched_parsed_text")
display(df_excel)

# COMMAND ----------

md_content_excel = df_excel.select("text").collect()[0].text
md(md_content_excel)

# COMMAND ----------

md_content_excel2 = df_excel.select("text").collect()[1].text
md(md_content_excel2)

# COMMAND ----------

# MAGIC %md
# MAGIC The table structured maintained better when parse from the excel files

# COMMAND ----------

# MAGIC %md
# MAGIC # Test Prompt Directly

# COMMAND ----------

from openai import OpenAI
import os

DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints"
)

# COMMAND ----------

def prompt_test(client, user_prompt):
    chat_completion = client.chat.completions.create(
        messages=[
        {
            "role": "system",
            "content": "You are an AI assistant"
        },
        {
            "role": "user",
            "content": user_prompt
        }
        ],
        model="databricks-claude-3-7-sonnet",
        max_tokens=256
    )

    return chat_completion.choices[0].message.content

# COMMAND ----------

# MAGIC %md
# MAGIC ## excel source

# COMMAND ----------

user_prompt = f"""Based on the information table
                <table>
                {md_content_excel2}
                </table>
                answer the following question: What kind of car accident does Mike Hughes experienced?"""

print(prompt_test(client, user_prompt))

# COMMAND ----------

user_prompt = f"""Based on the information table
                <table>
                {md_content_excel}
                </table>
                answer the following question: What is policy number of customer who lives in southeast Queens?"""

print(prompt_test(client, user_prompt))

# COMMAND ----------

# MAGIC %md
# MAGIC ## pdf source

# COMMAND ----------

user_prompt = f"""Based on the information table
                <table>
                {md_content_pdf}
                </table>
                answer the following question: What kind of car accident does Mike Hughes experienced?"""

print(prompt_test(client, user_prompt))

# COMMAND ----------

user_prompt = f"""Based on the information table
                <table>
                {md_content_pdf1}
                </table>
                answer the following question: What is policy number of customer who lives in southeast Queens?"""

print(prompt_test(client, user_prompt))

# COMMAND ----------

