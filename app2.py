import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import base64
import os
from thefuzz import process # Import for Fuzzy Search

# -----------------------------------------------------------------------------
# [HELPER FUNCTION] Convert Local Image to Base64 for HTML
# -----------------------------------------------------------------------------
def img_to_bytes(img_path):
    img_bytes = None
    if os.path.exists(img_path):
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            ext = img_path.split('.')[-1].lower()
            mime_type = "jpeg" if ext in ['jpg', 'jpeg'] else ext
            img_bytes = f"data:image/{mime_type};base64,{encoded_string}"
    return img_bytes

# -----------------------------------------------------------------------------
# [CONFIGURATION] Page Layout
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NITJ Research Admin",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# [CSS STYLING] EduAdmin Theme
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #F5F7FB; }
        [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E6EBF1; }
        
        .dashboard-card {
            background-color: #FFFFFF; padding: 25px; border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.03); margin-bottom: 20px; border: 1px solid #F0F2F5;
        }
        
        .welcome-banner {
            background: linear-gradient(120deg, #E3F2FD 0%, #FFFFFF 100%);
            border-left: 5px solid #1565C0; border-radius: 12px; padding: 25px; margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.02);
        }
        .welcome-banner h1 { color: #1565C0; font-size: 26px; font-weight: 700; margin: 0; }
        .welcome-banner p { color: #546E7A; margin: 5px 0 0 0; }

        div[data-testid="metric-container"] {
            background-color: white; border: 1px solid #E0E0E0; padding: 20px; border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02); transition: all 0.3s ease;
        }
        div[data-testid="metric-container"]:hover {
            border-color: #1565C0; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(21, 101, 192, 0.1);
        }
        div[data-testid="metric-container"] label { color: #78909C; font-size: 0.9rem; }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #263238; font-size: 1.8rem; font-weight: 700; }

        .user-row {
            display: flex; align-items: center; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid #F1F3F4;
        }
        .user-row:last-child { border-bottom: none; }
        .user-avatar {
            width: 40px; height: 40px; border-radius: 50%; background: #E3F2FD; color: #1565C0;
            display: flex; align-items: center; justify-content: center; font-weight: 700; margin-right: 15px;
        }
        .user-details { flex-grow: 1; }
        .user-name { font-weight: 600; color: #37474F; font-size: 0.95rem; }
        .user-dept { color: #90A4AE; font-size: 0.8rem; }
        .user-stat { font-weight: 700; color: #1565C0; }
        
        .podium-card { padding: 15px; border-radius: 12px; text-align: center; margin-bottom: 10px; }
        .podium-gold { border: 2px solid #FFD700; background: #FFFAF0; }
        .podium-silver { border: 2px solid #C0C0C0; background: #FAFAFA; }
        .podium-bronze { border: 2px solid #CD7F32; background: #FFF8F0; }

        .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
        .stTabs [data-baseweb="tab"] { height: 45px; border-radius: 8px; border: 1px solid transparent; background-color: #FFFFFF; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #1565C0; color: white; border: none; }

        .photo-container {
            text-align: center; margin-top: 20px; padding-top: 20px; border-top: 1px solid #F0F2F5;
        }
        .photo-avatar {
            width: 100px; height: 100px; border-radius: 50%; object-fit: cover;
            margin-bottom: 10px; border: 3px solid #E3F2FD;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .photo-name { font-weight: 600; color: #37474F; font-size: 1rem; }
        .photo-role { color: #90A4AE; font-size: 0.85rem; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# [DATA LOGIC]
# -----------------------------------------------------------------------------
def generate_dummy_data():
    departments = ['CSE', 'ECE', 'ME', 'Civil', 'Physics', 'Chem', 'Textile', 'IPE', 'ICE']
    names = [f"Dr. Faculty {i}" for i in range(1, 151)]
    # Adding Banalaxmi manually to dummy data for testing if no CSV
    names.append("Dr. Banalaxmi Brahma") 
    
    countries = ['India', 'USA', 'UK', 'Canada', 'Germany', 'Australia', 'Japan', 'France']
    weights = [0.6, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    data = []
    for name in names:
        dept = np.random.choice(departments)
        pubs = np.random.randint(5, 200)
        cites = int(pubs * np.random.uniform(5, 60)) 
        h_index = int(np.sqrt(cites) * np.random.uniform(0.8, 1.2))
        i10 = int(h_index * 1.5)
        country = np.random.choice(countries, p=weights)
        data.append({
            "name": name, "department": dept,
            "designation": np.random.choice(["Professor", "Associate Prof.", "Assistant Prof."]),
            "total_publications": pubs, "citations_box": cites,
            "h_index": h_index, "I10_Index": i10,
            "Research_Area": f"Specialization {np.random.randint(1, 10)}",
            "Country": country
        })
    return pd.DataFrame(data)

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

# -----------------------------------------------------------------------------
# [SIDEBAR & DATA MAPPING]
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# [DATA LOADING] Auto-load Default CSV + Optional Upload
# -----------------------------------------------------------------------------

st.sidebar.title("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Faculty Data (CSV)", type=["csv"])
df_final = None

if uploaded_file is not None:
    raw_df = load_csv(uploaded_file)
    st.sidebar.success("File uploaded!")
    cols = raw_df.columns.tolist()
    
    def get_idx(options, search):
        for i, opt in enumerate(options):
            if search.lower() in opt.lower(): return i
        return 0

    c_name = st.sidebar.selectbox("Name", cols, index=get_idx(cols, "name"))
    c_dept = st.sidebar.selectbox("Department", cols, index=get_idx(cols, "department"))
    c_desig = st.sidebar.selectbox("Designation", cols, index=get_idx(cols, "designation"))
    c_pubs = st.sidebar.selectbox("Publications", cols, index=get_idx(cols, "total_publications"))
    c_cite = st.sidebar.selectbox("Citations", cols, index=get_idx(cols, "citations_box"))
    c_h = st.sidebar.selectbox("H-Index", cols, index=get_idx(cols, "h_index"))
    
    try:
        df_final = raw_df.rename(columns={
            c_name: 'Name', c_dept: 'Department', c_desig: 'Designation',
            c_pubs: 'Total_Publications', c_cite: 'Total_Citations', c_h: 'H_Index'
        })
        for col in ['Total_Publications', 'Total_Citations', 'H_Index']:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0)
            
        if 'I10_Index' not in df_final.columns: df_final['I10_Index'] = df_final['H_Index']
        if 'Research_Area' not in df_final.columns: df_final['Research_Area'] = "General Engineering"
        if 'Country' not in df_final.columns:
            df_final['Country'] = np.random.choice(['India'], size=len(df_final))
        
    except Exception as e:
        st.error(f"Error mapping columns: {e}")

else:
    if st.sidebar.button("Load Demo Data"):
        df_final = generate_dummy_data()
        df_final = df_final.rename(columns={'name':'Name', 'department':'Department', 
                                          'designation':'Designation', 'total_publications':'Total_Publications',
                                          'citations_box':'Total_Citations', 'h_index':'H_Index'})

# -----------------------------------------------------------------------------
# [MAIN DASHBOARD]
# -----------------------------------------------------------------------------
if df_final is not None:
    depts = df_final['Department'].unique()
    sel_depts = st.sidebar.multiselect("Filter Department", depts, default=depts)
    filtered_df = df_final[df_final['Department'].isin(sel_depts)]

    st.markdown("""
        <div class="welcome-banner">
            <h1>🏛️ NIT Jalandhar | Research Dashboard</h1>
            <p>Analytics Portal: Track performance, citations, and global collaborations.</p>
        </div>
    """, unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["📊 Overview", "🏆 Leaderboard", "📈 Analytics", "👤 Faculty Profile"])

    # --- TAB 1: OVERVIEW ---
    with t1:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Faculty", len(filtered_df))
        kpi2.metric("Publications", f"{int(filtered_df['Total_Publications'].sum()):,}")
        kpi3.metric("Citations", f"{int(filtered_df['Total_Citations'].sum()):,}")
        kpi4.metric("Avg H-Index", f"{filtered_df['H_Index'].mean():.1f}")
        st.write("") 
        c_list, c_chart = st.columns([1.2, 1.8])
        with c_list:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("🌟 Top Contributors")
            top_n = filtered_df.sort_values(by="Total_Citations", ascending=False).head(5)
            for i, (idx, row) in enumerate(top_n.iterrows()):
                initials = "".join([x[0] for x in row['Name'].split()[:2]]).upper()
                st.markdown(f"""
                <div class="user-row">
                    <div style="display:flex; align-items:center;">
                        <div class="user-avatar">{initials}</div>
                        <div class="user-details">
                            <div class="user-name">{row['Name']}</div>
                            <div class="user-dept">{row['Designation']}</div>
                        </div>
                    </div>
                    <div class="user-stat">{int(row['Total_Citations'])}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c_chart:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Department Distribution")
            fig_pie = px.pie(filtered_df, names='Department', hole=0.6,
                             color_discrete_sequence=px.colors.qualitative.Prism)
            fig_pie.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 2: LEADERBOARD ---
    with t2:
        st.markdown("### 🏆 Research Impact Leaderboard")
        lb_df = filtered_df.copy()
        def min_max_scale(x): return (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
        lb_df['Score'] = ((min_max_scale(lb_df['Total_Citations'])*40) + 
                          (min_max_scale(lb_df['H_Index'])*40) + 
                          (min_max_scale(lb_df['Total_Publications'])*20)).round(1)
        lb_df = lb_df.sort_values('Score', ascending=False).reset_index(drop=True)
        top3 = lb_df.head(3)
        if len(top3) >= 3:
            c_gold, c_silver, c_bronze = st.columns(3)
            with c_gold:
                st.markdown(f"<div class='podium-card podium-gold'>🥇 <b>{top3.iloc[0]['Name']}</b><br>Score: {top3.iloc[0]['Score']}</div>", unsafe_allow_html=True)
            with c_silver:
                st.markdown(f"<div class='podium-card podium-silver'>🥈 <b>{top3.iloc[1]['Name']}</b><br>Score: {top3.iloc[1]['Score']}</div>", unsafe_allow_html=True)
            with c_bronze:
                st.markdown(f"<div class='podium-card podium-bronze'>🥉 <b>{top3.iloc[2]['Name']}</b><br>Score: {top3.iloc[2]['Score']}</div>", unsafe_allow_html=True)
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.dataframe(lb_df[['Name', 'Department', 'Total_Citations', 'H_Index', 'Score']],
                     use_container_width=True,
                     column_config={"Score": st.column_config.ProgressColumn("Impact Score", format="%.1f", min_value=0, max_value=100)})
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 3: ANALYTICS ---
    with t3:
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            dept_grp = filtered_df.groupby('Department')['Total_Publications'].mean().reset_index().sort_values('Total_Publications')
            fig_bar = px.bar(dept_grp, x='Total_Publications', y='Department', orientation='h',
                             title="Avg Pubs per Dept", color='Total_Publications', color_continuous_scale='Blues')
            fig_bar.update_layout(plot_bgcolor='white', xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with r1_c2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            fig_box = px.box(filtered_df, x='Department', y='H_Index', color='Department',
                             title="H-Index Distribution", points="outliers")
            fig_box.update_layout(showlegend=False, plot_bgcolor='white', xaxis_title=None)
            st.plotly_chart(fig_box, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        fig_scatter = px.scatter(filtered_df, x='Total_Publications', y='Total_Citations',
                                 size='H_Index', color='Department', hover_name='Name',
                                 log_x=True, log_y=True, title="Quantity vs Quality (Log Scale)", template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("🌍 Global Research Collaborations")
        map_data = filtered_df.groupby('Country').size().reset_index(name='Collaborations')
        fig_map = px.choropleth(map_data, locations="Country", locationmode='country names',
                                color="Collaborations", hover_name="Country", color_continuous_scale="Blues")
        fig_map.update_geos(showframe=False, showcoastlines=False, projection_type='natural earth')
        fig_map.update_layout(height=450, margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 4: FACULTY PROFILE ---
    with t4:
        c_search, c_profile = st.columns([1, 2])
        
        # [IMAGE LOADING LOGIC]
        director_img_code = img_to_bytes("director.jpg") 
        faculty_img_code = img_to_bytes("banalaxmi mam.jpeg")
        
        # Fallback
        if not director_img_code: director_img_code = "https://cdn-icons-png.flaticon.com/512/3135/3135768.png"
        if not faculty_img_code: faculty_img_code = "https://cdn-icons-png.flaticon.com/512/6833/6833605.png"

        with c_search:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### 🔍 Find Expert")
            
            # --- FUZZY SEARCH LOGIC ---
            all_names = filtered_df['Name'].tolist()
            txt = st.text_input("Search Name")
            
            # Default to all options
            match_options = all_names
            
            if txt:
                # Find the top 3 closest matches using fuzzy search
                matches = process.extract(txt, all_names, limit=3)
                # Matches format: [('Name1', score1), ('Name2', score2), ...]
                # We filter to keep matches with a score > 50 (to avoid garbage matches)
                match_options = [m[0] for m in matches if m[1] > 50]
                
                # If no good matches found, show nothing or all
                if not match_options:
                    st.warning("No close matches found. Try again.")
                    match_options = []

            selected = st.selectbox("Select Faculty", match_options)
            
            # --- PHOTO SECTION ---
            st.markdown(f"""
                <div class="photo-container">
                    <img src="{director_img_code}" class="photo-avatar">
                    <div class="photo-name">Director</div>
                    <div class="photo-role">NIT Jalandhar</div>
                </div>
                <div class="photo-container">
                    <img src="{faculty_img_code}" class="photo-avatar">
                    <div class="photo-name">Dr. Banalaxmi Brahma</div>
                    <div class="photo-role">Assistant Professor (Grade-II)</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c_profile:
            if selected:
                prof = filtered_df[filtered_df['Name'] == selected].iloc[0]
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.markdown(f"<h2 style='color:#003366;'>{prof['Name']}</h2><p>{prof['Designation']} | {prof['Department']}</p>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Citations", int(prof['Total_Citations']))
                m2.metric("H-Index", int(prof['H_Index']))
                m3.metric("Collaborations", prof['Country'])
                years = np.arange(2015, 2026)
                growth = np.cumsum(np.random.randint(1, int(prof['Total_Citations']/8)+2, size=11))
                growth = (growth / growth.max()) * prof['Total_Citations']
                fig_line = px.area(x=years, y=growth, title="Citation Growth")
                fig_line.update_traces(line_color='#1565C0', fill='tozeroy')
                fig_line.update_layout(plot_bgcolor='white')
                st.plotly_chart(fig_line, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👈 Please click 'Load Demo Data' in the sidebar.")
