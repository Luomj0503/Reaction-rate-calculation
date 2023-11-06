import numpy as np
import pandas as pd
import streamlit as st
import streamlit_option_menu
from streamlit_option_menu import option_menu
import base64
import textwrap

from texts import Texts
import pickle
import cirpy
import os

from PIL import Image
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
#from state import count_sessions
#count_sessions()

IMAGE_SUPP = Image.open('picture/3D_flow.png')
IMG_Fig1 = Image.open('picture/3D_flow.png')
IMG_Fig2 = Image.open('picture/O3.png')

class BackEnd:
    def __init__(self):
        self.kS_morgan_xgb = None
        self.kS_morgan_nn = None
        self.kS_morgan_rf = None
        self.OS_morgan_xgb = None
        self.OS_morgan_nn = None
        self.OS_morgan_rf = None
        self.ad = ApplicabilityDomain()
        self.base_train_kO3_morgan, self.base_train_kFeS_morgan, self.base_train_kO3_maccs, self.base_train_kFeS_maccs, self.base_train_kO3_both, self.base_train_kFeS_both = BackEnd.__load_basestrain(self)

        BackEnd.__load_models(self)

    #@st.cache_data
    def __load_models(self):
        self.kS_morgan_xgb = pickle.load(open(r'Models/FeS-1206XGB_mdl1.dat', 'rb'))
        self.kS_morgan_nn = pickle.load(open("Models/FeS-1206XGB_mdl2.dat", 'rb'))
        self.kS_morgan_rf= pickle.load(open("Models/FeS-1206XGB_mdl3.dat", 'rb'))
        self.OS_morgan_xgb = pickle.load(open(r'Models/O3 all new-1206XGB_mdl1.dat', 'rb'))
        self.OS_morgan_nn = pickle.load(open("Models/O3 all new-1206XGB_mdl1.dat", 'rb'))
        self.OS_morgan_rf= pickle.load(open("Models/O3 all new-1206XGB_mdl1.dat", 'rb'))

    

class FrontEnd(BackEnd):
    def __init__(self):
        super().__init__()
        gettext = Texts()
        self.text1 = gettext.text10()
        self.text1_2 = gettext.text10_2()
        self.text1_4 = gettext.text1_4()
        self.text2 = gettext.text2()
        self.text3 = gettext.text3()
        FrontEnd.main(self)

    def main(self):
        nav = FrontEnd.NavigationBar(self)

        # HOME
        if nav == 'HOME':
            st.header('Printability Under Different Printing Conditions by Python Simulator')
            st.markdown('{}'.format(self.text10), unsafe_allow_html=True)  # general description
            st.markdown('{}'.format(self.text10_2), unsafe_allow_html=True)  # The prediction of FeS
            st.image(IMG_Fig1)  # figure of 3D_flow
            col1, col2, col3 = st.columns([0.2, 5, 0.2])
        if nav == 'Printability':
            st.title('Simulation of Printability by Different Printing Conditions')
            Pressure = st.number_input('Choose Pressure(Pa)',0.0, 120.0)
            Speed = st.number_input('Choose Speed(mm/s)',0.0, 10.0)
            Nozzle_Diameter = st.number_input("Diameter of nozzle(mm)", 0, 0.9)
            Con = st.number_input("Concentration of Sodium alginate ink", 0.0, 0.2)
            cmodels = st.multiselect("Choose ML Models", ("XGBoost", "Neural Network", "Random Forest"),
                                     default="Neural Network")
            generate = st.button("Generate")
            if generate:
                for i in cmodels:
                    if i =="XGBoost":
                        np = []
                        feature_w_smiles = np.append([Pressure, Speed, Nozzle_Diameter, Con])
                        feature_w_smiles = feature_w_smiles.reshape(1, -1)
                        pred = self.3D_xgb.predict(feature_w_smiles)
                        if pred > = 0.9:
                            printability = 'excellent'
                        elif 0.9> pred > = 0.8:
                            printability = 'good'
                        elif 0.8> pred > = 0.7 :
                            printability = 'Ok'
                        elif 0.7> pred :
                            printability = 'bad'                            
                        st.markdown('## {}: {} '.format(i, printability),unsafe_allow_html=True)
                        
                    elif i =="Neural Network":
                        np = []
                        feature_w_smiles = np.append([Pressure, Speed, Nozzle_Diameter, Con])
                        feature_w_smiles = feature_w_smiles.reshape(1, -1)
                        pred = self.3D_xgb.predict(feature_w_smiles)
                        if pred > = 0.9:
                            printability = 'excellent'
                        elif 0.9> pred > = 0.8:
                            printability = 'good'
                        elif 0.8> pred > = 0.7 :
                            printability = 'Ok'
                        elif 0.7> pred :
                            printability = 'bad'
                        st.markdown('## {}: {} '.format(i, printability),unsafe_allow_html=True)

                    elif i =="Random Forest":
                        np = []
                        feature_w_smiles = np.append([Pressure, Speed, Nozzle_Diameter, Con])
                        feature_w_smiles = feature_w_smiles.reshape(1, -1)
                        pred = self.3D_xgb.predict(feature_w_smiles)
                        if pred > = 0.9:
                            printability = 'excellent'
                        elif 0.9> pred > = 0.8:
                            printability = 'good'
                        elif 0.8> pred > = 0.7 :
                            printability = 'Ok'
                        elif 0.7> pred :
                            printability = 'bad' 
                        st.markdown('## {}: {} '.format(i, printability),unsafe_allow_html=True)

                    
            
        if nav == 'About':
            st.markdown('{}'.format(self.text3), unsafe_allow_html=True)

        if nav == 'Citation':
            st.markdown('{}'.format(self.text1_4), unsafe_allow_html=True)

        if nav == 'Contact':
            # st.header(":mailbox: Entre em contato comigo!!")
            st.header("Contact me!!")
            contact_form = """
                    <form action="https://formsubmit.co/mphyschemlab@gmail.com" method="POST">
                     <input type="hidden" name="_captcha" value="false">
                     <input type="text" name="name" placeholder="Your name" optional>
                     <input type="email" name="email" placeholder="Your e-mail" optional>
                     <textarea name="message" placeholder="Type your message here"></textarea>
                     <button type="submit">Send</button>
                    </form>
                    """

            st.markdown(contact_form, unsafe_allow_html=True)
            FrontEnd.local_css("style.css")

    def local_css(file_name):
        with open(file_name) as f:
            return st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    def NavigationBar(self):
        with st.sidebar:
            nav = option_menu('Navegation:', ['HOME', 'Printability','About','Citation','Contact'],
                              icons=['house', 'book','box-arrow-in-left', 'journal-check',  'chat-left-text-fill'],
                              menu_icon="cast", default_index=0,styles={
                    "container": {"padding": "5!important", "background-color": "#fafafa"},"icon": {"color": "orange", "font-size": "25px"},
"nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},"nav-link-selected": {"background-color": "#02ab21"},})
            st.sidebar.markdown('# Contribute')
            st.sidebar.info('{}'.format(self.text2))
        return nav
if __name__ == '__main__':
    run = FrontEnd()

