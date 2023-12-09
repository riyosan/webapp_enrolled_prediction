import os
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import seaborn as sns
from PIL import Image
import streamlit as st
from dateutil import parser
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix, classification_report

with open("style.css") as f:
  styles = f.read()

# Tambahkan stylesheet ke aplikasi Streamlit
st.markdown(f'<style>{styles}</style>', unsafe_allow_html=True)

image=Image.open('logo.png')
# logo = st.columns((1.6, 0.7, 0.7))
home = st.container()
prepro = st.container()
train = st.container()
predict = st.container()
logo = home.columns((1.2, 1, 0.5))


#page layout
with logo[2]:
	st.image(image,width=160)
with logo[0]:
  st.title('**Tugas Akhir**')
  st.markdown('''*Penggunaan Algoritma Stacking Ensemble Learning Dalam Memprediksi Pengguna Enroll.*''')
  st.markdown('''**Riyo Santo Yosep - 171402020**''')

#####################
# Navigation
#####################
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark">
  <a class="navbar-brand" href="#" target="_blank">Riyo Santo Yosep</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="/">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#preprocessing">Preprocessing</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#training-testing">Training_Testing</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#predicting">Predicting</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)
#####################
# Function
#####################
@st.experimental_memo
def load_dataset(dataset):
  df = pd.read_csv(dataset)
  return df

@st.experimental_memo
def preprocessing(df):
  # df=load_dataset()
  #menghitung ulang isi dari kolom screen_litst karena tidak pas di kolom num screens
  df['screen_list'] = df.screen_list.astype(str) + ','
  df['num_screens'] = df.screen_list.astype(str).str.count(',')
  return df

@st.experimental_memo
def preprocessing_hour(df1):
    # df1 = preprocessing()
    # df1=df.copy()
    # feature engineering
    # karena kolom hour ada spasinya, maka kita ambil huruf ke 1 sampai ke 3
    df1['hour'] = df1['hour'].str.slice(1, 3).astype(int)

    # karena tipe data first_open dan enrolled_date itu adalah string, maka perlu diubah ke datetime
    df1['first_open'] = pd.to_datetime(df1['first_open'])
    # didalam dataset orang yg belum langganan itu NaN, maka jika i=string biarin, klo ga string diubah ke datetime kolom nan nya biarin tetap nat
    df1['enrolled_date'] = [parser.parse(i) if isinstance(i, str) else i for i in df1['enrolled_date']]

    # membuat kolom selisih , yaitu menghitung berapa lama orang yg first_open menjadi enrolled
    delta = (df1['enrolled_date'] - df1['first_open'])
    # convert timedelta to hours
    df1['selisih'] = delta / np.timedelta64(1, 'h')

    # karena di grafik menunjukkan orang kebanyakan enroll selama 24 jam pertama, maka kalau lebih dari 24 jam dianggap ga penting
    df1.loc[df1['selisih'] > 24, 'enrolled'] = 0

    return df1

@st.experimental_memo
def preprocessing_top_screens(df2):
  # df2=preprocessing_hour()
  # b=df2['screen_list'].apply(pd.Series).stack()
  # c = b.tolist()
  # from collections import Counter
  # p = Counter(' '.join(b).split()).most_common(100)
  # rslt = pd.DataFrame(p)
  # rslt.to_csv('data/top_screens.csv', index=False)
  top_screens=pd.read_csv('data/top_screens.csv')
  # diubah ke numppy arry dan mengambil kolom ke2 saja karna kolom1 isinya nomor
  top_screens=np.array(top_screens.loc[:,'top_screens'])
  df3 = df2.copy()
  #mengubah isi dari file top screen menjadi numerik
  for i in top_screens:
    df3[i]=df3.screen_list.str.contains(i).astype(int)
  #semua item yang ada di file top screen dihilangkan dari kolom screen list
  for i in top_screens:
    df3['screen_list']=df3.screen_list.astype(str).str.replace(i+',','')
  #menghitung jumlah item non top screen yang(tersisa) ada di screenlist
  df3['lainnya']=df3.screen_list.astype(str).str.count(',')
  return df3

@st.experimental_memo
def preprocessing_pred(df_pred):
  df=df_pred
  # df = preprocessing(df)
  #menghapus kolom numsreens yng lama
  df.drop(columns=['numscreens'], inplace=True)
  #mengubah kolom hour
  df.hour=df.hour.str.slice(1,3).astype(int)
  #karena tipe data first_open itu adalah string, maka perlu diubah ke datetime
  df.first_open=[parser.parse(i) for i in df.first_open]
  #import top_screen
  top_screens=pd.read_csv('top_screens.csv')
  top_screens=np.array(top_screens.loc[:,'top_screens'])
  for i in top_screens:
      df[i]=df.screen_list.str.contains(i).astype(int)
  for i in top_screens:
      df['screen_list']=df.screen_list.str.replace(i+',','')
  #menghitung jumlah item non top screen yang(tersisa) ada di screenlist
  df['lainnya']=df.screen_list.str.count(',')
  #menghapus double layar
  layar_loan = ['Loan','Loan2','Loan3','Loan4']
  df['jumlah_loan']=df[layar_loan].sum(axis=1)
  df.drop(columns=layar_loan, inplace=True)

  layar_saving = ['Saving1','Saving2','Saving2Amount','Saving4','Saving5','Saving6','Saving7','Saving8','Saving9','Saving10']
  df['jumlah_loan']=df[layar_saving].sum(axis=1)
  df.drop(columns=layar_saving, inplace=True)

  layar_credit = ['Credit1','Credit2','Credit3','Credit3Container','Credit3Dashboard']
  df['jumlah_credit']=df[layar_credit].sum(axis=1)
  df.drop(columns=layar_credit, inplace=True)

  layar_cc = ['CC1','CC1Category','CC3']
  df['jumlah_cc']=df[layar_cc].sum(axis=1)
  df.drop(columns=layar_cc, inplace=True)
  #mendefenisikan variabel numerik
  pred_numerik=df.drop(columns=['first_open','screen_list','user'], inplace=False)
  scaler = joblib.load('data/standar.joblib')
  fitur = pd.read_csv('data/fitur_pilihan.csv')
  fitur = fitur['0'].tolist()
  pred_numerik = pred_numerik[fitur]
  pred_numerik = scaler.transform(pred_numerik)
  model = joblib.load('data/stack_model.pkl')
  prediksi = model.predict(pred_numerik)
  prediksi2 = pd.DataFrame(prediksi)
  probabilitas = model.predict_proba(pred_numerik)
  user_id = df['user']
  prediksi_akhir = pd.Series(prediksi).rename('Pred',inplace=True)
  hasil_akhir= pd.concat([user_id,prediksi_akhir], axis=1).dropna()
  return probabilitas, hasil_akhir, prediksi2


@st.experimental_memo
def funneling(df3):
  # df=preprocessing_top_screens()
  #menggabungkan item yang mirip mirip, seperti kredit 1 kredit 2 dan kredit 3
  #funneling = menggabungkan beberapa screen yang sama dan menghapus layar yang sama
  layar_loan = ['Loan','Loan2','Loan3','Loan4']
  df3['jumlah_loan']=df3[layar_loan].sum(axis=1)
  df3.drop(columns=layar_loan, inplace=True)

  layar_saving = ['Saving1','Saving2','Saving2Amount','Saving4','Saving5','Saving6','Saving7','Saving8','Saving9','Saving10']
  df3['jumlah_loan']=df3[layar_saving].sum(axis=1)
  df3.drop(columns=layar_saving, inplace=True)

  layar_credit = ['Credit1','Credit2','Credit3','Credit3Container','Credit3Dashboard']
  df3['jumlah_credit']=df3[layar_credit].sum(axis=1)
  df3.drop(columns=layar_credit, inplace=True)

  layar_cc = ['CC1','CC1Category','CC3']
  df3['jumlah_cc']=df3[layar_cc].sum(axis=1)
  df3.drop(columns=layar_cc, inplace=True)
  #menghilangkan kolom yang ga relevan
  df_numerik=df3.drop(columns=['user','first_open','screen_list','enrolled_date','selisih','numscreens'], inplace=False)
  df_numerik.to_csv('data/df_numerik.csv', index=False)
  #determine the mutual information
  mutual_info = mutual_info_classif(df_numerik.drop(columns=['enrolled']), df_numerik.enrolled)
  mutual_info = pd.Series(mutual_info)
  mutual_info.index = df_numerik.drop(columns=['enrolled']).columns
  mutuals = mutual_info.sort_values(ascending=False)
  return df_numerik, mutuals

@st.experimental_memo
def choose_feature(df_numerik, jumlah_fitur):
  df=df_numerik.copy()
  mutual_info = mutual_info_classif(df.drop(columns=['enrolled']), df.enrolled)
  mutual_info = pd.Series(mutual_info)
  mutual_info.index = df.drop(columns=['enrolled']).columns
  mutual_info.sort_values(ascending=False)
  # from sklearn.feature_selection import SelectKBest
  fitur_terpilih = SelectKBest(mutual_info_classif, k = jumlah_fitur)
  fitur_terpilih.fit(df.drop(columns=['enrolled']), df.enrolled)
  pilhan_kolom = df.drop(columns=['enrolled']).columns[fitur_terpilih.get_support()]
  pd.Series(pilhan_kolom).to_csv('data/fitur_pilihan.csv',index=False)
  fitur = pilhan_kolom.tolist()
  fitur_baru = df[fitur]
  return fitur_baru

@st.experimental_memo
def standarization(fitur_baru):
  sc_X = StandardScaler()
  pilhan_kolom = sc_X.fit_transform(fitur_baru)
  joblib.dump(sc_X, 'data/standar.joblib')
  return pilhan_kolom

@st.experimental_memo
def split(df_numerik,pilhan_kolom, split_size):
  df=df_numerik.copy()
  X_train, X_test, y_train, y_test = train_test_split(pilhan_kolom, df['enrolled'],test_size=(100-split_size)/100, random_state=1)
  return X_train, X_test, y_train, y_test

@st.experimental_memo
def naive_bayes(X_train, X_test, y_train, y_test):
  nb = GaussianNB() # Define classifier)
  nb.fit(X_train, y_train)
  # Make predictions
  y_test_pred = nb.predict(X_test)
  matrik_nb = (classification_report(y_test, y_test_pred))
  cm_label_nb = pd.DataFrame(confusion_matrix(y_test, y_test_pred), columns=np.unique(y_test), index=np.unique(y_test))
  return matrik_nb, cm_label_nb, nb

@st.experimental_memo
def random_forest(X_train, X_test, y_train, y_test, parameter_n_estimators):
  rf = RandomForestClassifier(n_estimators=parameter_n_estimators,criterion="entropy", max_depth=2, random_state=1) # Define classifier
  rf.fit(X_train, y_train)
  # Make predictions
  y_test_pred = rf.predict(X_test)
  matrik_rf = (classification_report(y_test, y_test_pred))
  cm_label_rf = pd.DataFrame(confusion_matrix(y_test, y_test_pred), columns=np.unique(y_test), index=np.unique(y_test))
  return matrik_rf, cm_label_rf, rf


def stack_model(X_train, X_test, y_train, y_test, tetangga, nb, rf):
  # Build stack model
  estimator_list = [
      ('nb',nb),
      ('rf',rf)]
  stack_model = StackingClassifier(
      estimators=estimator_list, final_estimator=KNeighborsClassifier(tetangga, metric="euclidean"),cv=5
  )
  # Train stacked model
  stack_model.fit(X_train, y_train)
  # Make predictions
  y_test_pred = stack_model.predict(X_test)
  # Evaluate model
  matrik_stack = (classification_report(y_test, y_test_pred))
  cm_label_stack = pd.DataFrame(confusion_matrix(y_test, y_test_pred), columns=np.unique(y_test), index=np.unique(y_test))
  joblib.dump(stack_model, 'data/stack_model.pkl')
  return matrik_stack, cm_label_stack,y_test_pred

def load_sample_dataset():
  # Gantilah dengan cara memuat dataset contoh Anda
  sample_data = pd.read_csv('data/fintech_data.csv')
  df = pd.DataFrame(sample_data)
  return df
#####################
# Preprocess
#####################
prepro.markdown('''
## Preprocessing
''')
# with st.expander("sebelum mulai training"):
with st.sidebar.header('1. Preprocess'):
  options = ["Upload your dataset", "Use Sample Dataset"]
  selected_option = st.sidebar.selectbox("Select Dataset Option", options)

# Load dataset based on user's choice
  dataset = None

  if selected_option == "Upload your dataset":
    dataset = st.sidebar.file_uploader("Upload your dataset", type=["csv"], key="dataset")
    if dataset is not None:
        df = pd.read_csv(dataset)
        st.sidebar.success("Dataset loaded successfully!")
  elif selected_option == "Use Sample Dataset":
      dataset = 'data/fintech_data.csv' # Gantilah dengan nama file sampel Anda
      df = pd.read_csv(dataset)
      st.sidebar.success("Using Sample Dataset")
expander = prepro.expander(
  "Data preprocessing adalah teknik awal data mining untuk mengubah data mentah menjadi format dan informasi yang lebih efisien dan bermanfaat.")
expander.markdown(" ")
if dataset is not None:
  df=load_dataset(dataset)
  expander.subheader('Fintech Dataset')
  expander.write(df)
  expander.markdown("""---""")
  expander.subheader('Revisi numscreens')
  df1=preprocessing(df)
  container = expander.columns((1.9, 1.1))
  expander.caption('merevisi nilai yang ada pada kolom numscreens.')
  expander.markdown("""---""")
  df1_types = df1.dtypes.astype(str)
  with container[0]:
    st.write(df1)
  with container[1]:
    st.write(df1_types)

  expander.subheader('Revisi hour')
  df2=preprocessing_hour(df1)
  container1 = expander.columns((1.9, 1.1))
  expander.caption('merevisi isi kolom hour dan merubah tipe data ke numerik.')
  expander.markdown("""---""")
  df2_types = df2.dtypes.astype(str)
  with container1[0]:
    st.write(df2)
  with container1[1]:
    st.write(df2_types)

  expander.subheader('One-hot encoding')
  df3=preprocessing_top_screens(df2)
  expander.write(df3)
  expander.caption('membuat kolom baru berdasarkan isi kolom screen_list.')
  expander.markdown("""---""")

  expander.subheader('Hasil akhir')
  df_numerik, mutuals = funneling(df3)
  expander.write(df_numerik)
  expander.caption('menghapus kolom-kolom yang tidak penting.')
  mutuals.sort_values(ascending=False).plot.bar(title='Korelasi tiap kolom terhadap keputusan enrolled')
  st.set_option('deprecation.showPyplotGlobalUse', False)
  expander.pyplot()
  expander.caption('mengurutkan korelasi setiap kolom terhadap kelasnya(enrolled)')
  expander.markdown("""---""")
else:
  expander.warning('Upload Dataset Pada Sidebar No.1')

#####################
# Train_test
#####################

train.markdown('''
## Training_Testing
''')
expander2 = train.expander(
  "Training adalah proses melatih algoritma agar mengenali pola dari suatu dataset, Testing adalah proses evaluasi terhadap performa algoritma yang sudah di latih.")
expander2.markdown(" ")
with st.sidebar.header('2. Set Parameter'):
  split_size = st.sidebar.slider('Rasio Pembagian Data (% Untuk Data Latih)', 10, 90, 80, 5)
  jumlah_fitur = st.sidebar.slider('jumlah pilihan fitur (Untuk Data Latih)', 5, 47, 20, 5)
  parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 10, 100, 50, 10)
  tetangga = st.sidebar.slider('Jumlah K (KNN)', 11, 101, 55, 11)

if "load_state" not in st.session_state:
  st.session_state.load_state = False
if st.sidebar.button('Latih & Uji') or st.session_state.load_state:
  st.session_state.load_state = True
  df_numerik = pd.read_csv('data/df_numerik.csv')
#   df_numerik = funneling(df3)
  fitur_baru = choose_feature(df_numerik, jumlah_fitur)
  pilhan_kolom=standarization(fitur_baru)
  X_train, X_test, y_train, y_test = split(df_numerik,pilhan_kolom, split_size)
  matrik_nb, cm_label_nb, nb = naive_bayes(X_train, X_test, y_train, y_test)
  matrik_rf, cm_label_rf, rf = random_forest(X_train, X_test, y_train, y_test, parameter_n_estimators)
  matrik_stack, cm_label_stack, y_test_pred = stack_model(X_train, X_test, y_train, y_test, tetangga, nb, rf)

  nb_container = expander2.columns((0.8, 1))
  #page layout
  with nb_container[0]:
    st.write("2a. Naive Bayes report using sklearn")
    st.text('Naive Bayes Report:\n ' + matrik_nb)
  # st.write(" ")
  # st.write(" ")
  # st.write(" ")
  with nb_container[1]:
    cm_label_nb.index.name = 'Actual'
    cm_label_nb.columns.name = 'Predicted'
    sns.heatmap(cm_label_nb, annot=True, cmap='Blues', fmt='g')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
  # st.write(" ")
  # st.write(" ")

  # Evaluate model
  rf_container = expander2.columns((0.8, 1))
  #page layout
  with rf_container[0]:
    st.write("2b. Random Forest report using sklearn")
    st.text('Random Forest Report:\n ' + matrik_rf)
  # st.write(" ")
  # st.write(" ")
  # st.write(" ")
  with rf_container[1]:
    cm_label_rf.index.name = 'Actual'
    cm_label_rf.columns.name = 'Predicted'
    sns.heatmap(cm_label_rf, annot=True, cmap='Blues', fmt='g')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
  # st.write(" ")
  # st.write(" ")

  stack_container = expander2.columns((0.8, 1))
  #page layout
  with stack_container[0]:
    st.write("2c. Stack report using sklearn")
    st.text('Stack Report:\n ' + matrik_stack)
  # st.write(" ")
  # st.write(" ")
  # st.write(" ")

  with stack_container[1]:
    cm_label_stack.index.name = 'Actual'
    cm_label_stack.columns.name = 'Predicted'
    sns.heatmap(cm_label_stack, annot=True, cmap='Blues', fmt='g')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
  st.write(" ")  

  #take df3 from apps/praproses.py
  try:
    df1 = df3
    var_enrolled = df1['enrolled']
    # #membagi menjadi train dan test untuk mencari user id
    X_train, X_test, y_train, y_test = train_test_split(df1, df1['enrolled'], test_size=(100-split_size)/100, random_state=1)
    train_id = X_train['user']
    test_id = X_test['user']
    #menggabungkan semua
    y_pred_series = pd.Series(y_test).rename('Aktual',inplace=True)
    hasil_akhir = pd.concat([y_pred_series, test_id], axis=1).dropna()
    hasil_akhir['Prediksi']=y_test_pred
    hasil_akhir = hasil_akhir[['user','Aktual','Prediksi']].reset_index(drop=True)
    container_hasil_akhir = expander2.columns((0.8, 1.4, 0.8))
    with container_hasil_akhir[1]:
      st.text('Tabel Perbandingan Asli dan Prediksi:\n ')
      st.dataframe(hasil_akhir)
  except:
    expander2.error('Please do preprocessing first')
else:
  expander2.warning('Latih & Uji Terlebih Dahulu Pada Sidebar No.2')
#####################
# Predict
#####################
predict.markdown('''
## Predicting
''')
expander3 = predict.expander(
  "Predicting adalah tahapan untuk menerapkan model yang sudah dilatih dan divalidasi, untuk membuat prediksi berdasarkan dataset baru yang sebelumnya belum pernah dikenali oleh model.")
expander3.markdown(" ")
st.sidebar.write(" ")
st.sidebar.write(" ")
with st.sidebar.header('3. Predict'):
  data_pred = st.sidebar.file_uploader("Upload your file",type=['csv'], key="data_pred")
data_pred = st.session_state.data_pred
if data_pred is not None:
  dataset = data_pred
  df = load_dataset(dataset)
  expander3.subheader('Predict Dataset')
  expander3.dataframe(df)
  df_pred = preprocessing(df)
  if st.sidebar.button("Predict Data"):
    hasil_akhir, probabilitas, prediksi2 = preprocessing_pred(df_pred)
    layout = expander3.columns((1,1,1,1))
    with layout[1]:
      st.subheader('Probabilitas')
      st.write(hasil_akhir)
    with layout[2]:
      st.subheader('Prediksi')
      st.write(probabilitas)
    prediksi2['Enrolled']=prediksi2
    prediksi3 = prediksi2[['Enrolled']].reset_index(drop=True)
    jumlah = prediksi3.groupby('Enrolled').size()
    jumlah.plot.bar(color="blue")
    expander3.subheader('Visualisasi Perbandingan Prediksi')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    expander3.pyplot()
else:
  expander3.warning('Upload Dataset Yang Ingin di Prediksi Pada Sidebar No.3')
