import streamlit as st
import random
from g4f.client import Client

# Initialize AI Client
client = Client()

def analyze_influencer(name):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Analyze the influence of {name} and generate credibility, longevity, and engagement scores. Also, provide a short description of the influencer."}],
        web_search=False
    )
    return response.choices[0].message.content

def get_real_top_influencers(field=None, limit=10):
    """
    Returns real influencer data based on field category.
    
    Args:
        field (str, optional): The field/category to filter by
        limit (int, optional): Maximum number of influencers to return
        
    Returns:
        list: A list of influencer dictionaries
    """
    real_influencers = {
        # Tech Entrepreneurs
        'tech_entrepreneur': [
            {
                'platform': 'twitter',
                'handle': 'elonmusk',
                'name': 'Elon Musk',
                'field': 'tech_entrepreneur',
                'overall_score': 9.5,
                'component_scores': {
                    'credibility': 9.2,
                    'longevity': 8.8,
                    'engagement': 9.8
                },
                'profile_image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Elon_Musk_Royal_Society_%28crop2%29.jpg/800px-Elon_Musk_Royal_Society_%28crop2%29.jpg'
            },
            {
                'platform': 'twitter',
                'handle': 'pmarca',
                'name': 'Marc Andreessen',
                'field': 'tech_entrepreneur',
                'overall_score': 8.9,
                'component_scores': {
                    'credibility': 9.0,
                    'longevity': 9.1,
                    'engagement': 8.5
                },
                'profile_image_url': 'https://upload.wikimedia.org/wikipedia/commons/7/7b/Marc_Andreessen_2023.png'
            },
            {
                'platform': 'twitter',
                'handle': 'naval',
                'name': 'Naval Ravikant',
                'field': 'tech_entrepreneur',
                'overall_score': 8.8,
                'component_scores': {
                    'credibility': 9.1,
                    'longevity': 8.4,
                    'engagement': 8.7
                },
                'profile_image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Naval_Ravikant_at_Blockcon_2018_%28cropped%29.jpeg/800px-Naval_Ravikant_at_Blockcon_2018_%28cropped%29.jpeg'
            }
        ],
        
        # Content Creators
        'content_creator': [
            {
                'platform': 'youtube',
                'handle': 'MrBeast',
                'name': 'Jimmy Donaldson',
                'field': 'content_creator',
                'overall_score': 9.4,
                'component_scores': {
                    'credibility': 8.9,
                    'longevity': 9.2,
                    'engagement': 9.8
                },
                'profile_image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/MrBeast_at_CCXP_%28cropped%29.png/800px-MrBeast_at_CCXP_%28cropped%29.png'
            },
            {
                'platform': 'instagram',
                'handle': 'emmachamberlain',
                'name': 'Emma Chamberlain',
                'field': 'content_creator',
                'overall_score': 8.7,
                'component_scores': {
                    'credibility': 8.3,
                    'longevity': 8.5,
                    'engagement': 9.2
                },
                'profile_image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Emma_Chamberlain_2019_by_Glenn_Francis.jpg/800px-Emma_Chamberlain_2019_by_Glenn_Francis.jpg'
            },
            {
                'platform': 'tiktok',
                'handle': 'charlidamelio',
                'name': 'Charli D\'Amelio',
                'field': 'content_creator',
                'overall_score': 8.6,
                'component_scores': {
                    'credibility': 7.9,
                    'longevity': 8.2,
                    'engagement': 9.6
                },
                'profile_image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/Charli_D%27Amelio_in_September_2023.jpg/800px-Charli_D%27Amelio_in_September_2023.jpg'
            }
        ],
        
        # Athletes
        'athlete': [
            {
                'platform': 'instagram',
                'handle': 'cristiano',
                'name': 'Cristiano Ronaldo',
                'field': 'athlete',
                'overall_score': 9.7,
                'component_scores': {
                    'credibility': 9.5,
                    'longevity': 9.8,
                    'engagement': 9.8
                },
                'profile_image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Cristiano_Ronaldo_playing_for_Al_Nassr_FC_against_Persepolis%2C_September_2023_%28cropped%29.jpg/800px-Cristiano_Ronaldo_playing_for_Al_Nassr_FC_against_Persepolis%2C_September_2023_%28cropped%29.jpg'
            },
            {
                'platform': 'instagram',
                'handle': 'leomessi',
                'name': 'Lionel Messi',
                'field': 'athlete',
                'overall_score': 9.6,
                'component_scores': {
                    'credibility': 9.6,
                    'longevity': 9.7,
                    'engagement': 9.5
                },
                'profile_image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Lionel_Messi_20180626.jpg/800px-Lionel_Messi_20180626.jpg'
            },
            {
                'platform': 'instagram',
                'handle': 'kingjames',
                'name': 'LeBron James',
                'field': 'athlete',
                'overall_score': 9.5,
                'component_scores': {
                    'credibility': 9.3,
                    'longevity': 9.6,
                    'engagement': 9.5
                },
                'profile_image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/LeBron_James_crop.jpg/800px-LeBron_James_crop.jpg'
            }
        ],
        
        # Musicians
        'musician': [
            {
                'platform': 'instagram',
                'handle': 'taylorswift',
                'name': 'Taylor Swift',
                'field': 'musician',
                'overall_score': 9.6,
                'component_scores': {
                    'credibility': 9.3,
                    'longevity': 9.7,
                    'engagement': 9.8
                },
                'profile_image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Taylor_Swift_at_the_2023_MTV_Video_Music_Awards.jpg/800px-Taylor_Swift_at_the_2023_MTV_Video_Music_Awards.jpg'
            },
            {
                'platform': 'instagram',
                'handle': 'badgalriri',
                'name': 'Rihanna',
                'field': 'musician',
                'overall_score': 9.3,
                'component_scores': {
                    'credibility': 9.1,
                    'longevity': 9.5,
                    'engagement': 9.2
                },
                'profile_image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Rihanna_Fenty_2018.png/800px-Rihanna_Fenty_2018.png'
            },
            {
                'platform': 'instagram',
                'handle': 'drake',
                'name': 'Drake',
                'field': 'musician',
                'overall_score': 9.2,
                'component_scores': {
                    'credibility': 8.8,
                    'longevity': 9.3,
                    'engagement': 9.4
                },
                'profile_image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Drake_July_2016.jpg/800px-Drake_July_2016.jpg'
            }
        ],
        
        # Academics
        'academic': [
            {
                'platform': 'twitter',
                'handle': 'neiltyson',
                'name': 'Neil deGrasse Tyson',
                'field': 'academic',
                'overall_score': 9.2,
                'component_scores': {
                    'credibility': 9.7,
                    'longevity': 8.9,
                    'engagement': 8.8
                },
                'profile_image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Neil_deGrasse_Tyson_2017.jpg/800px-Neil_deGrasse_Tyson_2017.jpg'
            },
            {
                'platform': 'twitter',
                'handle': 'BrianCox',
                'name': 'Brian Cox',
                'field': 'academic',
                'overall_score': 8.9,
                'component_scores': {
                    'credibility': 9.5,
                    'longevity': 8.7,
                    'engagement': 8.4
                },
                'profile_image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/Brian_Cox_at_the_Science_Museum%2C_London%2C_October_2019.jpg/800px-Brian_Cox_at_the_Science_Museum%2C_London%2C_October_2019.jpg'
            },
            {
                'platform': 'twitter',
                'handle': 'karensnyc',
                'name': 'Karen Nyberg',
                'field': 'academic',
                'overall_score': 8.7,
                'component_scores': {
                    'credibility': 9.6,
                    'longevity': 8.3,
                    'engagement': 8.0
                },
                'profile_image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Karen_L._Nyberg.jpg/800px-Karen_L._Nyberg.jpg'
            }
        ]
    }

    all_influencers = []
    for category_influencers in real_influencers.values():
        all_influencers.extend(category_influencers)

    all_influencers = sorted(all_influencers, key=lambda x: x['overall_score'], reverse=True)

    if field:
        if field == "all":
            results = all_influencers[:limit]
        else:
            results = real_influencers.get(field, [])[:limit]
    else:
        results = all_influencers[:limit]
    
    return results

def generate_score():
    return round(random.uniform(7.5, 9.9), 1)  # Mock score range

def display_influencer_card(influencer):
    """
    Display an influencer card with their image and key metrics
    """
    st.markdown(
        f"""
        <div style="text-align: center; border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <img src="{influencer.get('profile_image_url', '')}" style="width: 100px; height: 100px; border-radius: 50%;" alt="{influencer.get('name', 'Unknown')}">
            <div style="font-size: 24px; font-weight: bold;">{influencer.get('name', 'Unknown')}</div>
            <div style="font-size: 16px; color: #777;">@{influencer.get('handle', '')}</div>
            <div style="font-size: 32px; font-weight: bold; margin: 10px 0;">{influencer.get('overall_score', 0)}</div>
            <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                <div>
                    <div style="font-weight: bold;">{influencer.get('component_scores', {}).get('credibility', 0)}</div>
                    <div>Credibility</div>
                </div>
                <div>
                    <div style="font-weight: bold;">{influencer.get('component_scores', {}).get('longevity', 0)}</div>
                    <div>Longevity</div>
                </div>
                <div>
                    <div style="font-weight: bold;">{influencer.get('component_scores', {}).get('engagement', 0)}</div>
                    <div>Engagement</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Set page configuration
st.set_page_config(page_title="Infame", layout="wide")

# Sidebar
st.sidebar.title("Categories")
categories = ["All Influencers", "Tech Entrepreneurs", "Content Creators", "Athletes", "Musicians", "Academics"]
selected_category = st.sidebar.radio("", categories)

# Mapping for categories to field values
field_mapping = {
    "All Influencers": "all",
    "Tech Entrepreneurs": "tech_entrepreneur",
    "Content Creators": "content_creator",
    "Athletes": "athlete",
    "Musicians": "musician",
    "Academics": "academic"
}

selected_field = field_mapping.get(selected_category, "all")

st.sidebar.title("Filters")
st.sidebar.checkbox("Rising Stars")
st.sidebar.checkbox("Legacy Figures")

# App header
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>Infame</h1>", unsafe_allow_html=True)

# Manual Search Section
st.subheader("Search for an Influencer")
search_query = st.text_input("Enter influencer's name:")
# Modified part for the search functionality in your Streamlit app
# Modified part for the search functionality in your Streamlit app
if st.button("Search") and search_query:
    st.write("Fetching real-time data...")
    influencer_description = analyze_influencer(search_query)
    
    # Generate mock scores for the component metrics
    credibility_score = generate_score()
    longevity_score = generate_score()
    engagement_score = generate_score()
    overall_score = round((credibility_score + longevity_score + engagement_score) / 3, 1)
    
    st.markdown(
        f"""
        <div style="text-align: center; border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <div style="font-size: 24px; color: #1f77b4;">🖼️</div>
            <div style="font-size: 24px; font-weight: bold;">{search_query}</div>
            <div style="font-size: 32px; font-weight: bold; margin: 10px 0;">{overall_score}</div>
            <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                <div>
                    <div style="font-weight: bold;">{credibility_score}</div>
                    <div>Credibility</div>
                </div>
                <div>
                    <div style="font-weight: bold;">{longevity_score}</div>
                    <div>Longevity</div>
                </div>
                <div>
                    <div style="font-weight: bold;">{engagement_score}</div>
                    <div>Engagement</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Display the AI-generated description separately
    st.subheader("About this Influencer")
    st.write(influencer_description)

# Main content - Top Influencers
st.header(f"Top {selected_category}")

# Get real influencer data
influencers = get_real_top_influencers(field=selected_field, limit=3)

# Display influencers in a grid
if influencers:
    cols = st.columns(min(3, len(influencers)))
    for i, influencer in enumerate(influencers[:3]):
        with cols[i]:
            display_influencer_card(influencer)
else:
    st.info("No influencers found")

st.header("Analyze New Influencer")
col1, col2 = st.columns(2)

with col1:
    new_platform = st.selectbox("Platform", ["Twitter", "Reddit", "Instagram", "YouTube", "TikTok"])

with col2:
    new_handle = st.text_input("Handle", placeholder="Enter username without @")

new_field = st.selectbox("Field", [
    "Tech Entrepreneur", 
    "Content Creator", 
    "Athlete", 
    "Musician", 
    "Academic"
])

if st.button("Analyze"):
    if new_handle:
        field_value = new_field.lower().replace(" ", "_")
        platform_value = new_platform.lower()
        
        with st.spinner(f"Analyzing {new_handle}'s influence..."):
            all_influencers = get_real_top_influencers(limit=100)
            found_influencer = next((i for i in all_influencers if i['handle'].lower() == new_handle.lower() and i['platform'].lower() == platform_value.lower()), None)
            
            if found_influencer:
                result = found_influencer
            else:
                # Create a mock result if not found
                result = {
                    'platform': platform_value,
                    'handle': new_handle,
                    'name': new_handle.capitalize(),
                    'field': field_value,
                    'overall_score': generate_score(),
                    'component_scores': {
                        'credibility': generate_score(),
                        'longevity': generate_score(),
                        'engagement': generate_score()
                    },
                    'profile_image_url': f'https://via.placeholder.com/150?text={new_handle[0].upper()}'
                }
        
        if result:
            st.success("Analysis complete!")
            display_influencer_card(result)

            st.subheader("Detailed Metrics")
            
            component_scores = result.get('component_scores', {})
            
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric("Credibility", f"{component_scores.get('credibility', 0):.1f}/10")
            with metrics_cols[1]:
                st.metric("Longevity", f"{component_scores.get('longevity', 0):.1f}/10")
            with metrics_cols[2]:
                st.metric("Engagement", f"{component_scores.get('engagement', 0):.1f}/10")

            st.subheader("Influence Trend")
            
            trend_data = {
                "Influence Score": [7.5, 7.8, 8.2, 8.5, 8.9, result.get('overall_score', 9.0)]
            }
            
            st.line_chart(trend_data)

            st.subheader("Competitor Comparison")

            competitors = [inf for inf in get_real_top_influencers(field=field_value, limit=5) 
                          if inf['handle'].lower() != new_handle.lower()][:3]
            
            if competitors:
                comp_data = {
                    result.get('name', 'Current'): result.get('overall_score', 0)
                }
                
                for comp in competitors:
                    comp_data[comp.get('name', 'Unknown')] = comp.get('overall_score', 0)
                
                st.bar_chart(comp_data)
            else:
                st.info("No competitors found for comparison")
        else:
            st.error(f"Could not retrieve data for {new_handle} on {platform_value}")
    else:
        st.warning("Please enter a handle to analyze")

st.markdown("---")
st.markdown("Infame - Measuring Real Influence")