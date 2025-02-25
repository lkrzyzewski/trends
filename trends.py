import os
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser, StructuredOutputParser, ResponseSchema
from pydantic import BaseModel, Field
from typing import List
import mysql.connector as mariadb
import matplotlib.pyplot as plt
import pandas as pd
import json
import logging
import sys

os.environ["OPENAI_API_KEY"] = st.secrets["OpenAI_key"]
os.environ["GOOGLE_API_KEY"]  = st.secrets["Gemini_key"]

dbuser = st.secrets["dbuser"]
dbpass = st.secrets["dbpass"]
dbhost = st.secrets["dbhost"]
dbdb = st.secrets["dbdb"]

st_auth_file = "auth.yaml"
upload_dir = "files/"

with open(st_auth_file) as file:
    authconf = yaml.load(file, Loader=SafeLoader)
    
authenticator = stauth.Authenticate(
    authconf['credentials'],
    authconf['cookie']['name'],
    authconf['cookie']['key'],
    authconf['cookie']['expiry_days']
)

authenticator.login('main')

if 'catname' not in st.session_state: st.session_state.catname = None
if 'newcatname' not in st.session_state: st.session_state.newcatname = ""
if 'clicked' not in st.session_state: st.session_state.clicked = False
if 'sheet' not in st.session_state: st.session_state.sheet = ""
if 'titlecol' not in st.session_state: st.session_state.titlecol = ""
if 'desccol' not in st.session_state: st.session_state.desccol = ""
if 'categories' not in st.session_state: st.session_state.categories = ""
if 'catvisibility' not in st.session_state: st.session_state.catvisibility = False
if 'kategorie' not in st.session_state: st.session_state.kategorie = {}
if 'cattodel' not in st.session_state: st.session_state.cattodel = None
if 'airunclicked' not in st.session_state: st.session_state.airunclicked = False
if 'dataprep' not in st.session_state: st.session_state.dataprep = False
if 'batchsize' not in st.session_state: st.session_state.batchsize = 100

def set_catname(catname):  st.session_state.cats = catname
def set_catname(newcatname):  st.session_state.newcatname = newcatname
def set_titlecol(titlecolname): st.session_state.titlecol = titlecolname
def set_desccol(desccolname): st.session_state.desccol = desccolname
def set_sheet(sheetname): st.session_state.sheet = sheetname
def set_clicked(): st.session_state.clicked = True
def set_airunclicked(): st.session_state.airunclicked = True
def set_categories(categories):  st.session_state.categories = categories
def set_batchsize(size):  st.session_state.batchsize = size

@st.cache_resource
def configure_logging(file_path, level=logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def cat_add():
    if st.session_state.newcatname not in st.session_state.kategorie:
        try:
            mariadb_connection = mariadb.connect(user=dbuser, password=dbpass,
                                                        host=dbhost,
                                                        database=dbdb)

            mariadb_connection.autocommit = True
            cursor = mariadb_connection.cursor(buffered=True, dictionary=True)

            cursor.execute("INSERT INTO tredy_kategorie (name, categories) VALUES (%s, %s)", (st.session_state.newcatname, st.session_state.categories))

            cursor.close()
            mariadb_connection.close()
        except Exception as e:
            print(e)
        finally:
            st.session_state.kategorie[st.session_state.newcatname] = st.session_state.categories
            st.session_state.newcatname = ""
            st.session_state.categories = ""
            logger.info(f"Add category: {st.session_state.newcatname} with data: {st.session_state.categories}")
def cat_del():
    try:
        mariadb_connection = mariadb.connect(user=dbuser, password=dbpass,
                                                        host=dbhost,
                                                        database=dbdb)

        mariadb_connection.autocommit = True
        cursor = mariadb_connection.cursor(buffered=True, dictionary=True)

        cursor.execute("DELETE FROM tredy_kategorie WHERE name=%s", (st.session_state.cattodel,))

        cursor.close()
        mariadb_connection.close()
    except Exception as e:
        print(e)
    finally:
        st.session_state.kategorie.pop(st.session_state.cattodel)
        logger.info(f"Delete category: {st.session_state.cattodel}")


#chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", transport="grpc", temperature=0)
#chat_model = ChatOpenAI(temperature=0, model="gpt-4o")

class category(BaseModel):
    category:str = Field(description="Category of article")
    subject:str = Field(descriptions="The subject of a category, for example, if the category is a vegetable, what kind of vegetable is it or if it is a car, what make and model is it") 
    
class categories(BaseModel):
    categories: List[category] = Field(description="List of categories of articles. Each of category corresponds to article in input list.")
    
def promptllm(inputdata, examples, chat_model):
    try:
        output_parser = PydanticOutputParser(pydantic_object=categories)

        prompt = PromptTemplate(
            template = "Answer the question as best you can in Polish. \n{format_instructions}\n{question}",
            input_variables=["questions"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()},
        )

        document_prompt = f"""Your task is to Extract categories from list of article titles and short snippets. 
        Create a list of extracted categories and it's subjects which corresponds to article list.
        Don't create categories that are too general, but also try not to have too many of them. 
        These are sample categories that you can use, or you can create new ones that are more appropriate.: {examples} 
        List of categories needs to be in the same order and exact quantity as list of articles.

        {inputdata}
        """

        _input = prompt.format_prompt(question=document_prompt)  
        chain = prompt | chat_model | output_parser 

        data = chain.invoke({"question": _input})
        return data
    except  Exception as e:
        print(e)
        return False
def steps_count(rowcount, batchsize):
    if rowcount > int(rowcount/batchsize)*batchsize:
        return int(rowcount/batchsize) +1
    else:
        return int(rowcount/batchsize) 
    
def examples_to_list(examples):
    if len(examples) > 0:
        if "," in examples:
            examples_list = examples.split(",")
            for i, _ in enumerate(examples_list):
                examples_list[i] = examples_list[i].strip()
            return examples_list
        else:
            example_list = [ examples ]
            return example_list

def list_to_examples(list):
    if len(list) > 0:
        examples = ""
        for item in list:
            examples += f"{item}, "
        return examples[:-2]

@st.fragment    
def show_download_button(filename, finalfile):
    st.download_button(label="**Pobierz plik excel**", use_container_width=True, 
                            data=finalfile,
                            file_name=f"trendy_{filename}",
                            mime='application/octet-stream')
    pass
        
if st.session_state["authentication_status"]:
    logger = configure_logging('tredy.log')
    #st.session_state.kategorie["smaki"] ="desery, dania mięsne, warzywa, zupy, owoce, dania zdrowe, napoje , promocje"
    mariadb_connection = mariadb.connect(user=dbuser, password=dbpass,
                                                        host=dbhost,
                                                        database=dbdb)

    mariadb_connection.autocommit = True
    cursor = mariadb_connection.cursor(buffered=True, dictionary=True)

    cursor.execute("SELECT * FROM tredy_kategorie")
    if cursor.rowcount:
        for row in cursor:
            st.session_state.kategorie[row['name']] = row['categories'] 
    cursor.close()
    mariadb_connection.close()
    
    with st.sidebar:
        st.radio("Wybierz model językowy", ["Gemini2-flash", "Gemini2-Pro-exp","GPT4o"], key="model")
        st.divider()
        st.header("W celu lepszego dopasowania podaj kilka przykładowych kategorii")
        st.divider()
        if len(st.session_state.kategorie) != 0:
            with st.container(height=250, border=True):
                category_sel = st.selectbox("Wybież zapisane kategorie", list(st.session_state.kategorie.keys()), index=None, key='catname')
                if st.session_state.catname in st.session_state.kategorie:
                    st.write(st.session_state.kategorie[st.session_state.catname])
        st.divider()
        with st.expander("Dodaj kategorię", expanded=False):
            kategorie_txt = st.text_area("Przykładowe kategorie", value=st.session_state.categories, placeholder=st.session_state.categories, 
                                        label_visibility='visible', height=200, disabled=st.session_state.catvisibility, key="categories")
            kategorie_name = st.text_input("Nazwa", value=st.session_state.newcatname, placeholder=st.session_state.newcatname,
                                        label_visibility='visible', disabled=st.session_state.catvisibility, key="newcatname")
            st.button("Dodaj kategorię", on_click=cat_add)
        st.divider()
        with st.expander("Usuń kategorię", expanded=False):
            if len(st.session_state.kategorie) != 0:
                category_to_del = st.selectbox("Wybież kategorię do usunięcia", list(st.session_state.kategorie.keys()), index=None, key='cattodel')
                st.button("Usuń kategorię", on_click=cat_del)
        st.divider()
    
    st.header(f'*Kategorie - Trendy*', divider='blue')
    uploaded_file = st.file_uploader("Wybież plik excel", accept_multiple_files=False)
    
    st.button('Upload Files', on_click=set_clicked)
    if st.session_state.clicked and uploaded_file:
        bytes_data = uploaded_file.read()
        filepath = os.path.join(upload_dir, uploaded_file.name)        
        with(open(filepath, "wb")) as f:
            f.write(bytes_data)
            logger.info(f"File uploaded: {uploaded_file.name}")
        with st.status(label="✅ **Plik pobrany...**", state="running", expanded=True) as status:
        
            sheets = (pd.ExcelFile(filepath)).sheet_names
            if len(sheets) > 1:
                sheetbox = st.selectbox("Wybież arkusz", sheets, index=None, key='sheet')
            else:
                set_sheet(sheets[0])

            if st.session_state.sheet:
                logger.info(f"Chosen Sheet: {st.session_state.sheet}")
                status.update(label="✅ **Arkusz wybrany...**")
                data = pd.read_excel(filepath, sheet_name=st.session_state.sheet)
                columns = data.columns.values.tolist()
                col1, col2 = st.columns(2)
                with col1:
                    titlebox = st.selectbox("Wybież kolumnę z tytułem", columns, index=None, key='titlecol')
                with col2:
                    descbox = st.selectbox("Wybież kolumnę z opisem", columns, index=None, key='desccol')

                if st.session_state.titlecol and st.session_state.desccol and st.session_state.titlecol != st.session_state.desccol:
                    logger.info(f"Tittle column: {st.session_state.titlecol}, Snippet collumn: {st.session_state.desccol}")
                    status.update(label="✅ **Kolumny wybrane...**")
                    titles = pd.DataFrame
                    titles = data[[st.session_state.titlecol, st.session_state.desccol]].copy()
                    st.dataframe(titles.head(10), use_container_width=True, height=300)
 
                    if st.session_state.catname:
                        st.button('**Wyodrębnij kategorie**', on_click=set_airunclicked, use_container_width=True)
                        if st.session_state.airunclicked:
                            status.update(label="✅ **Dane przygotowane...**", state="complete", expanded=False)
                            st.session_state.dataprep = True
        if st.session_state.dataprep:
            with st.status("🤖 **AI pracuje...**", state="running", expanded=True) as aistatus:
                if st.session_state["model"] == "GPT4o":
                    chat_model = ChatOpenAI(temperature=0, model="gpt-4o")
                    set_batchsize(100)
                elif st.session_state["model"] == "Gemini2-flash":
                    chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", transport="grpc", temperature=0)
                    set_batchsize(200)
                elif st.session_state["model"] == "Gemini2-Pro-exp":
                    chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05", transport="grpc", temperature=0)
                    set_batchsize(200)
                logger.info(f"Choosen model: {st.session_state.model}")
                data = pd.read_excel(filepath, sheet_name=st.session_state.sheet)
                
                titles = pd.DataFrame
                titles = data[['title', 'snippet']].copy()
                batchsize = st.session_state.batchsize
                rowcount = titles.shape[0]
                if batchsize > rowcount:
                    batchsize = rowcount
                    set_batchsize(rowcount)
                aidata = []
                steps = steps_count(rowcount, batchsize)
                step = 1 / steps
                percent_complete = 0
                progress_text = "Postęp"
                logger.info(f"----AI Processing...  Data row count: {rowcount}, batch size: {batchsize}, Steps count {steps}")
                
                error = True
                start_examples = st.session_state.kategorie[st.session_state.catname]
                example_list = examples_to_list(start_examples)
                maxretry = 2
                retry=1
                i=0
                progress_bar = st.progress(percent_complete, text=progress_text)
                while i < int(rowcount/batchsize):
                    inputdata = json.dumps(json.loads(titles[i*batchsize:(i+1)*batchsize].to_json(orient="records")), indent=3, ensure_ascii=False)
                    with st.spinner(f"Partia {i+1} z {steps}", show_time=True):
                        bachdata = promptllm(inputdata, list_to_examples(example_list), chat_model)
                        if bachdata:
                            if len(bachdata.categories) == batchsize:
                                logger.info(f"Batch{i+1}: Data: {len(bachdata.categories)}, Batchsize: {batchsize}")
                                for cat in bachdata.categories:
                                    aidata.append({'category': cat.category, 'subject': cat.subject})
                                    if cat.category not in example_list:
                                        example_list.append(cat.category)
                                logger.info(f"Batch{i+1}: Categories: {list_to_examples(example_list)}")
                                i+=1
                                retry=1
                                progress_bar.progress(percent_complete + step, text=progress_text)
                                percent_complete += step
                            else:
                                logger.info(f"Retry batch{i+1}: Data: {len(bachdata.categories)}, Batchsize: {batchsize}")
                                if retry == maxretry:
                                    logger.info(f"Max retry reached. Batch{i+1}: Data: {len(bachdata.categories)}, Batchsize: {batchsize}")
                                    st.error(f"Ilość kategorii nie zgadza się: Data: {len(bachdata.categories)}, Batchsize: {batchsize}")
                                    error = False
                                    break
                            retry += 1

                if rowcount > int(rowcount/batchsize)*batchsize and error:
                    retry=1
                    i=0
                    while i<1:
                        inputdata = json.dumps(json.loads(titles[int(rowcount/batchsize)*batchsize:rowcount].to_json(orient="records")), indent=3, ensure_ascii=False)
                        with st.spinner(f"Partia {steps-1} z {steps}", show_time=True):
                            bachdata = promptllm(inputdata, list_to_examples(example_list), chat_model)
                            if bachdata:
                                if len(bachdata.categories) == rowcount - int(rowcount/batchsize)*batchsize:
                                    logger.info(f"Batch{steps}: Data: {len(bachdata.categories)}, Batchsize: {batchsize}")
                                    for cat in bachdata.categories:
                                        aidata.append({'category': cat.category, 'subject': cat.subject})
                                    logger.info(f"Batch{steps}: Categories: {list_to_examples(example_list)}")
                                    i+=1
                                    retry=1
                                    progress_bar.progress(percent_complete + step, text=progress_text)
                                    percent_complete += step

                                else:
                                    logger.info(f"Retry batch{steps}: Data: {len(bachdata.categories)}, Batchsize: {batchsize}")
                                    if retry == maxretry:
                                        logger.info(f"Max retry reached. Batch{i+1}: Data: {len(bachdata.categories)}, Batchsize: {batchsize}")
                                        st.error(f"Ilość kategorii nie zgadza się: Data: {len(bachdata.categories)}, Batchsize: {batchsize}")
                                        break
                                    retry += 1
                    
               
                aistatus.update(label="✅ Kategorie wyodrębnione", state="complete", expanded=True)
                logger.info(f"----AI Processing end. File rows: {rowcount}, AI Rows {len(aidata)}")
                
                st.write(f"Wiersze plik: {rowcount}, Wiersze ai: {len(aidata)}")
                if rowcount == len(aidata): 
                    catcount = {}

                    for index, cat in enumerate(aidata):
                        if cat['category'] not in catcount:
                            catcount[cat['category']] = 1
                        else:
                            catcount[cat['category']] += 1

                    catcount = dict(sorted(catcount.items(), key=lambda item: item[1], reverse=True))
                    catcount = {k: catcount[k] for k in list(catcount)[:10]}
                    
                    labels = [label for label in catcount]
                    counts = [catcount[count] for count in catcount]
                    fig, ax = plt.subplots()
                    plt.xticks(rotation=90)
                    plt.title("Kategorie główne")
                    plt.ylabel("Ilość")
                    ax.bar(labels, counts)
                    
                    st.pyplot(fig)
                    
                    aidataframe = pd.DataFrame(aidata)
                    finaldata = pd.concat([aidataframe, data], axis=1)
                    finaldata.to_excel(f"trendy_{uploaded_file.name}")
                    
                    with open(f"trendy_{uploaded_file.name}", "rb") as finalxlsx:
                        finalfile = finalxlsx.read()
                        
                        show_download_button(uploaded_file.name, finalfile)
                

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')