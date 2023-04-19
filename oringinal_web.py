import numpy as np
import pandas as pd
import streamlit as st
import streamlit_option_menu
from streamlit_option_menu import option_menu
import base64
import textwrap
from Similarity_calculation import ApplicabilityDomain

from texts import Texts
import pickle
import cirpy
import os
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdDepictor, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
#from state import count_sessions
#count_sessions()

IMAGE_SUPP = Image.open('picture/FeS.png')
IMG_Fig1 = Image.open('picture/FeS.png')
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

    def __load_basestrain(self):
        kO3_morgan = 'Similarity_calculation/MF_O3.csv'
        kFeS_morgan = 'Similarity_calculation/MF_FeS.csv'
        kO3_maccs = 'Similarity_calculation/MACCS_O3.csv'
        kFeS_maccs = 'Similarity_calculation/MACCS_FeS.csv'
        kO3_both = 'Similarity_calculation/Both_O3.csv'
        kFeS_both = 'Similarity_calculation/Both_FeS.csv'

        self.base_train_kO3_morgan = pd.read_csv(kO3_morgan).values
        self.base_train_kFeS_morgan = pd.read_csv(kFeS_morgan).values
        self.base_train_kO3_maccs = pd.read_csv(kO3_maccs).values
        self.base_train_kFeS_maccs = pd.read_csv(kFeS_maccs).values
        self.base_train_kO3_both = pd.read_csv(kO3_both).values
        self.base_train_kFeS_both = pd.read_csv(kFeS_both).values
        return self.base_train_kO3_morgan, self.base_train_kFeS_morgan, self.base_train_kO3_maccs, self.base_train_kFeS_maccs, self.base_train_kO3_both, self.base_train_kFeS_both

    def _applicabilitydomain(self, data, typefp: str, radical: str):
        if typefp == 'morgan':
            if radical == 'kFeS':
                get_simdf = self.ad.analyze_similarity(base_test=data, base_train=self.base_train_kFeS_morgan)
                similiraty = get_simdf['Max'].values
                return similiraty

            elif radical == 'kO3':
                get_simdf = self.ad.analyze_similarity(base_test=data, base_train=self.base_train_kO3_morgan)
                similiraty = get_simdf['Max'].values
                return similiraty

        elif typefp == 'maccs':
            if radical == 'kFeS':
                get_simdf = self.ad.analyze_similarity(base_test=data, base_train=self.base_train_kFeS_maccs)
                similiraty = get_simdf['Max'].values
                return similiraty

            elif radical == 'kO3':
                get_simdf = self.ad.analyze_similarity(base_test=data, base_train=self.base_train_kOH_maccs)
                similiraty = get_simdf['Max'].values
                return similiraty

        elif typefp == 'both':
            if radical == 'kFeS':
                get_simdf = self.ad.analyze_similarity(base_test=data, base_train=self.base_train_kFeS_both)
                similiraty = get_simdf['Max'].values
                return similiraty

            elif radical == 'kO3':
                get_simdf = self.ad.analyze_similarity(base_test=data, base_train=self.base_train_kO3_both)
                similiraty = get_simdf['Max'].values
                return similiraty

    
    def __moltosvg(self, mol, molSize=(320, 320), kekulize=True):
        mol = Chem.MolFromSmiles(mol)
        pkl = pickle.dumps(mol)
        mc = pickle.loads(pkl)
        #mc = Chem.Mol(mol.ToBinary())
        if kekulize:
            try:
                Chem.Kekulize(mc)
            except:
                mc = pickle.loads(pkl)
                #mc = Chem.Mol(mol.ToBinary())
        if not mc.GetNumConformers():
            rdDepictor.Compute2DCoords(mc)
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
        drawer.DrawMolecule(mc)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg.replace('svg:', '')

    def _render_svg(self, smiles):
        svg = BackEnd.__moltosvg(self, mol=smiles)
        b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
        html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
        st.write(html, unsafe_allow_html=True)

    def _makeMorganFingerPrint(self, smiles, nbits: int, raio=2):
        mol = Chem.MolFromSmiles(smiles)
        bi = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, nBits=nbits, radius=raio, bitInfo=bi)
        fp = np.array(fp)
        return fp, bi

    def _makeMaccsFingerprint(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        fps = MACCSkeys.GenMACCSKeys(mol)
        fps = np.array(fps)
        return fps


class FrontEnd(BackEnd):
    def __init__(self):
        super().__init__()
        gettext = Texts()
        self.text1 = gettext.text1()
        self.text1_2 = gettext.text1_2()
        self.text1_3 = gettext.text1_3()
        self.text1_4 = gettext.text1_4()
        self.text2 = gettext.text2()
        self.text3 = gettext.text3()
        FrontEnd.main(self)

    def main(self):
        nav = FrontEnd.NavigationBar(self)

        # HOME
        if nav == 'HOME':
            st.header('Python Simulator of Rate Constant')
            st.markdown('{}'.format(self.text1), unsafe_allow_html=True)  # general description
            st.markdown('{}'.format(self.text1_2), unsafe_allow_html=True)  # The prediction of FeS
            st.image(IMG_Fig1)  # figure of FeS
            st.markdown('{}'.format(self.text1_3), unsafe_allow_html=True)  # The prediction of O3
            col1, col2, col3 = st.columns([0.2, 5, 0.2])
            col2.image(IMG_Fig2, use_column_width=True)  # figure of O3
        if nav == 'S-ZVI Reaction Rate Simulation':
            st.title('Simulation of reaction rate between S-ZVI and organic pollutants')
            smi_casrn = st.text_input('Type SMILES or CAS Number', 'C(=C(Cl)Cl)Cl')
            # test casnumber or smiles
            if smi_casrn.count('-') == 2:  # change CAS Number into SMILES
                casrn2smi = cirpy.resolve(smi_casrn, 'smiles')
                # st.write(casrn2smi) #to show smiles of casrn
                smi_casrn = casrn2smi
            else:
                pass

            show_molecule = st.button('Show')
            if show_molecule:
                show = st.button('Hide')
                FrontEnd._render_svg(self, smi_casrn)  # plot molecule
            pH = st.number_input('Choose pH(3~8)',3.0, 8.0)
            T = st.number_input('Choose T(20~50)',20.0, 50.0)
            FeS_con = st.number_input("Concentration of S-ZVI(g/L)", 0.0, 8.0)
            S_Fe = st.number_input("Ratio of Sulfur Content to Iron Content", 0.0, 0.4)
            Cod = st.number_input("Concentration of Organic pollutant(mol/L)", 0.0, 8.0)
            fprints = st.radio("Choose type molecular fingerprint", ('Morgan', 'MACCS', 'both'))
            cmodels = st.multiselect("Choose ML Models", ("XGBoost", "Neural Network", "Random Forest"),
                                     default="Neural Network")
            generate = st.button("Generate")
            if generate:
                if fprints =="Morgan":
                    for i in cmodels:
                        if i =="XGBoost":
                            fp, frags = FrontEnd._makeMorganFingerPrint(self, smiles=smi_casrn, nbits=2048, raio=2)
                            fp = fp.reshape(1, -1)
                            feature_w_smiles = np.append(fp, [pH, T, Cod, S_Fe, FeS_con])
                            feature_w_smiles = feature_w_smiles.reshape(1, -1)
                            pred = self.kS_morgan_xgb.predict(feature_w_smiles)
                            st.markdown('## {}: {} h<sup>-1'.format(i, pred),unsafe_allow_html=True)
                        elif i =="Neural Network":
                            fp, frags = FrontEnd._makeMorganFingerPrint(self, smiles=smi_casrn, nbits=2048, raio=2)
                            fp = fp.reshape(1, -1)
                            feature_w_smiles = np.append(fp,[pH, T, Cod, S_Fe, FeS_con])
                            feature_w_smiles = feature_w_smiles.reshape(1,-1)
                            pred = self.kS_morgan_nn.predict(feature_w_smiles)
                            st.markdown('## {}: {} h<sup>-1'.format(i, pred),unsafe_allow_html=True)
                        elif i =="Random Forest":
                            fp, frags = FrontEnd._makeMorganFingerPrint(self, smiles=smi_casrn, nbits=2048, raio=2)
                            fp = fp.reshape(1, -1)
                            feature_w_smiles = np.append(fp, [pH, T, Cod, S_Fe, FeS_con])
                            feature_w_smiles = feature_w_smiles.reshape(1, -1)
                            pred = self.kS_morgan_rf.predict(feature_w_smiles)
                            st.markdown('## {}: {} h<sup>-1'.format(i, pred),unsafe_allow_html=True)
                        # calc AD
                        sim = FrontEnd._applicabilitydomain(self, data=fp, typefp='morgan',radical='kFeS')
                        st.markdown('<font color="green">The molecule is with the applicability domain. ({}% Similarity)</font>'.format(
                                (sim * 100).round(2)), unsafe_allow_html=True)

        if nav == 'O3 Reaction Rate Simulation':
            st.title('Simulation of reaction rate between O3 and organic pollutants')
            smi_casrn = st.text_input('Type SMILES or CAS Number', 'C(=C(Cl)Cl)Cl')
            # test casnumber or smiles
            if smi_casrn.count('-') == 2:  # change CAS Number into SMILES
                casrn2smi = cirpy.resolve(smi_casrn, 'smiles')
                # st.write(casrn2smi) #to show smiles of casrn
                smi_casrn = casrn2smi
            else:
                pass

            show_molecule = st.button('Show')
            if show_molecule:
                show = st.button('Hide')
                # st.image(FrontEnd._mol2img(self, smi_casrn))
                FrontEnd._render_svg(self, smi_casrn)  # plot molecule
            pH = st.number_input('Choose pH(3~8)',3.0, 8.0)
            T = st.number_input('Choose T(20~50)',20.0, 50.0)
            O3_con = st.number_input("Concentration of O3(g/L)", 0.0, 8.0)
            Cod = st.number_input("Concentration of Organic pollutant(mol/L)",0.0,8.0)
            fprints = st.radio("Choose type molecular fingerprint", ('Morgan', 'MACCS', 'both'))


            cmodels = st.multiselect("Choose ML Models", ("XGBoost", "Neural Network", "Random Forest"),
                                     default="Neural Network")
            generate = st.button("Generate")
            if generate:
                if fprints =="Morgan":
                    for i in cmodels:
                        if i =="XGBoost":
                            fp, frags = FrontEnd._makeMorganFingerPrint(self, smiles=smi_casrn, nbits=2048, raio=2)
                            fp = fp.reshape(1, -1)
                            feature_w_smiles = np.append(fp, [pH, T, Cod, O3_con])
                            feature_w_smiles = feature_w_smiles.reshape(1, -1)
                            pred = self.OS_morgan_xgb.predict(feature_w_smiles)[0]
                            st.markdown('## {}: {} h<sup>-1'.format(i, pred),unsafe_allow_html=True)
                        elif i =="Neural Network":
                            fp, frags = FrontEnd._makeMorganFingerPrint(self, smiles=smi_casrn, nbits=2048, raio=2)
                            fp = fp.reshape(1, -1)
                            feature_w_smiles = np.append(fp,[pH, T, Cod, O3_con])
                            feature_w_smiles = feature_w_smiles.reshape(1,-1)
                            pred = self.OS_morgan_nn.predict(feature_w_smiles)[0]
                            st.markdown('## {}: {} h<sup>-1'.format(i, pred),unsafe_allow_html=True)
                        elif i =="Random Forest":
                            fp, frags = FrontEnd._makeMorganFingerPrint(self, smiles=smi_casrn, nbits=2048, raio=2)
                            fp = fp.reshape(1, -1)
                            feature_w_smiles = np.append(fp, [pH, T, Cod, O3_con])
                            feature_w_smiles = feature_w_smiles.reshape(1, -1)
                            pred = self.OS_morgan_rf.predict(feature_w_smiles)[0]
                            st.markdown('## {}: {} h<sup>-1'.format(i, pred),unsafe_allow_html=True)
                         # calc AD
                        sim = FrontEnd._applicabilitydomain(self, data=fp, typefp='morgan',radical='kO3')
                        st.markdown('<font color="green">The molecule is with the applicability domain. ({}% Similarity)</font>'.format(
                                (sim * 100).round(2)), unsafe_allow_html=True)

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
            nav = option_menu('Navegation:', ['HOME', 'S-ZVI Reaction Rate Simulation','O3 Reaction Rate Simulation','About','Citation','Contact'],
                              icons=['house', 'water', 'book','box-arrow-in-left', 'journal-check',  'chat-left-text-fill'],
                              menu_icon="cast", default_index=0,styles={
                    "container": {"padding": "5!important", "background-color": "#fafafa"},"icon": {"color": "orange", "font-size": "25px"},
"nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},"nav-link-selected": {"background-color": "#02ab21"},})
            st.sidebar.markdown('# Contribute')
            st.sidebar.info('{}'.format(self.text2))
        return nav
if __name__ == '__main__':
    run = FrontEnd()

