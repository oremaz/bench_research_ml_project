import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import streamlit as st
import json
import hashlib
from pathlib import Path

# --- Ensure all session state keys are initialized ---
def initialize_session_state():
    """Initialize all session state keys to prevent KeyError."""
    keys_to_init = {
        'user_info': {},
        'initialized': False,
        'is_new_user': True,
        'username': None,
        'password': None,
        'agent_graph': None,
        'food_predictor': None,
        'chat_session_id': None,  # Unique ID for current chat session
        'selected_memory_chats': [],  # List of chat session IDs to include as memory
        'conversation_state': {
            "messages": [],
            "user_profile": {},
            "weight_goal": {},
            "daily_constraints": {},
            "previous_results": {},
            "meal_plan": {},
            "nutritional_values": {},
            "profile_setup_complete": False
        },
        'chat_history': [],
        'show_recipe_analyzer': False,
        'show_daily_results': False,
        'show_modify_dialog': False,
        'show_register': False
    }
    
    for key, default_value in keys_to_init.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# Import modules that might access session state
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, date
import json
from improved_agent import build_nutritionist_graph, initialize_components
from model_predictor import FoodModelPredictor

# Page configuration
st.set_page_config(
    page_title="AI Nutritionist Agent",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #000 !important;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
        color: #000 !important;
    }
    .assistant-message {
        background-color: #f1f8e9;
        margin-right: 2rem;
        color: #000 !important;
    }
</style>
""", unsafe_allow_html=True)

def init_agent():
    """Initialize the nutritionist agent."""
    if st.session_state['agent_graph'] is None:
        # Try to load API key from .env if not set
        import os
        api_key = st.sidebar.text_input("Google API Key", type="password", key="api_key")
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            st.sidebar.info("Loaded API key from environment.")
        if api_key:
            try:
                st.sidebar.info("Initializing agent...")
                initialize_components(api_key)
                # Use streamlit_mode=True to bypass human input node
                st.session_state['agent_graph'] = build_nutritionist_graph(api_key, streamlit_mode=True)
                st.sidebar.success("✅ Agent initialized successfully!")
                return True
            except Exception as e:
                st.sidebar.error(f"❌ Error initializing agent: {e}")
                return False
        else:
            st.sidebar.warning("Please enter your Google API Key to continue")
            return False
    return True

def display_user_profile():
    """Display user profile information in the sidebar."""
    profile = st.session_state['conversation_state'].get("user_profile", {})
    
    # Also check if we have user registration info
    username = st.session_state.get('username')
    user_info = None
    if username:
        user_info = load_user_info(username)
    
    if (profile and st.session_state['conversation_state'].get("profile_setup_complete", False)) or user_info:
        st.sidebar.subheader("👤 Your Profile")
        
        # Show registration info if available
        if user_info:
            with st.sidebar.expander("📋 Profile Summary", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Age", f"{user_info.get('age', 'N/A')}")
                    st.metric("Weight", f"{user_info.get('weight', 'N/A')} kg")
                with col2:
                    st.metric("Height", f"{user_info.get('height', 'N/A')} cm")
                    if user_info.get('weight') and user_info.get('height'):
                        bmi = user_info['weight'] / ((user_info['height']/100) ** 2)
                        st.metric("BMI", f"{bmi:.1f}")
                
                st.write("**Goal:** " + user_info.get('primary_goal', 'Not specified'))
                st.write("**Activity:** " + user_info.get('activity_level', 'Not specified')[:20] + "...")
                
                # Show meal schedule summary
                meal_schedule = user_info.get('meal_schedule', {})
                if meal_schedule:
                    enabled_meals = []
                    for meal_name, meal_info in meal_schedule.items():
                        if meal_info.get('enabled'):
                            if meal_name == 'snacks':
                                enabled_meals.append(f"{meal_name} ({meal_info.get('frequency', 'N/A')})")
                            else:
                                enabled_meals.append(f"{meal_name} ({meal_info.get('location', 'N/A')})")
                    if enabled_meals:
                        st.write("**Meals:** " + ", ".join(enabled_meals[:2]) + ("..." if len(enabled_meals) > 2 else ""))
                
                if user_info.get('dietary_preferences'):
                    st.write("**Diet:** " + ", ".join(user_info['dietary_preferences'][:2]))
        
        # Show conversation profile if available
        if profile:
            with st.sidebar.expander("🤖 AI Analysis", expanded=False):
                if profile.get("registration_complete"):
                    st.success("✅ Registration data active in AI")
                    st.write("**AI has access to:**")
                    st.write("• Health & dietary info")
                    st.write("• Goals & preferences") 
                    st.write("• Lifestyle constraints")
                    st.write("• Food preferences")
                
                conversation_context = profile.get("conversation_context", "")
                llm_understanding = profile.get("llm_understanding", "")
                if conversation_context:
                    st.text_area("Context:", conversation_context, height=60, disabled=True)
                if llm_understanding:
                    st.text_area("AI Understanding:", llm_understanding, height=100, disabled=True)
        
        # Quick stats
        meal_plan = st.session_state['conversation_state'].get("meal_plan", {})
        if meal_plan:
            st.sidebar.metric("🍽️ Meal Plan", "Ready" if meal_plan else "Not Set")
    else:
        st.sidebar.info("👋 Chat with the AI to set up your nutrition profile!")

def display_nutrition_dashboard():
    """Display nutrition dashboard with charts."""
    st.subheader("📊 Nutrition Dashboard")
    
    # Sample data for demonstration
    # In a real app, this would come from the user's actual data
    sample_data = {
        'Date': [datetime.now().date() for _ in range(7)],
        'Calories': [2000, 1950, 2100, 1900, 2050, 1980, 2020],
        'Protein': [120, 115, 130, 110, 125, 118, 122],
        'Carbs': [250, 240, 260, 230, 255, 245, 250],
        'Fat': [65, 62, 68, 60, 66, 63, 65]
    }
    
    df = pd.DataFrame(sample_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calories over time
        fig_calories = px.line(df, x='Date', y='Calories', 
                              title='Daily Calories Intake',
                              color_discrete_sequence=['#2E8B57'])
        fig_calories.update_layout(height=300)
        st.plotly_chart(fig_calories, use_container_width=True)
    
    with col2:
        # Macronutrient breakdown for today
        macros = [df.iloc[-1]['Protein'] * 4, df.iloc[-1]['Carbs'] * 4, df.iloc[-1]['Fat'] * 9]
        labels = ['Protein', 'Carbs', 'Fat']
        
        fig_pie = px.pie(values=macros, names=labels, 
                        title='Today\'s Macronutrient Breakdown',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

def display_meal_plan():
    """Display the current meal plan."""
    meal_plan = st.session_state['conversation_state'].get("meal_plan", {})
    
    if meal_plan and meal_plan.get("plan"):
        st.subheader("🍽️ Today's Meal Plan")
        
        # Display the meal plan
        plan_content = meal_plan["plan"] if "plan" in meal_plan else meal_plan.get("complete_plan", "")
        
        # Try to structure the meal plan if it's raw text
        st.markdown(plan_content)
        
        # Add buttons for meal actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Regenerate Plan"):
                # Clear current meal plan to trigger regeneration
                st.session_state['conversation_state']["meal_plan"] = {}
                st.rerun()
        
        with col2:
            if st.button("📝 Modify Plan"):
                st.session_state['show_modify_dialog'] = True
        
        with col3:
            if st.button("✅ Confirm Plan"):
                st.success("Meal plan confirmed!")
    else:
        st.info("No meal plan generated yet. Chat with the assistant to create one!")

def display_chat_interface():
    """Display the main chat interface."""
    st.subheader("💬 Chat with Your AI Nutritionist")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, (role, message) in enumerate(st.session_state['chat_history']):
            if role == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message"><strong>AI Nutritionist:</strong> {message}</div>', 
                           unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input and st.session_state['agent_graph']:
        # Add user message to chat history immediately
        st.session_state['chat_history'].append(("user", user_input))
        
        try:
            # Capture session state values we need
            agent_graph = st.session_state['agent_graph']
            conversation_state = st.session_state['conversation_state'].copy()
            
            from langchain_core.messages import HumanMessage
            current_messages = conversation_state.get("messages", [])
            
            # Get context from selected previous chats
            username = st.session_state.get('username')
            selected_memory_chats = st.session_state.get('selected_memory_chats', [])
            
            enhanced_user_input = user_input
            if username and selected_memory_chats:
                memory_context = get_selected_chat_context(username, selected_memory_chats)
                if memory_context:
                    enhanced_user_input = f"{memory_context}\n\nCURRENT USER MESSAGE: {user_input}"
                    st.sidebar.info(f"🧠 Added context from {len(selected_memory_chats)} previous chat(s)")
            
            conversation_state["messages"] = current_messages + [HumanMessage(content=enhanced_user_input)]
            
            # Debug info
            st.sidebar.write(f"🔍 Processing message #{len(current_messages) + 1}")
            
            # Direct invocation with progress feedback
            with st.spinner("AI Nutritionist is thinking..."):
                try:
                    config = {"recursion_limit": 20}
                    result = agent_graph.invoke(conversation_state, config)
                    
                    if result is None:
                        st.error("Agent returned no result.")
                        st.session_state['chat_history'].append(("assistant", "I'm having trouble processing your request. Please try again."))
                        st.rerun()
                        return
                        
                    # Update session state with results
                    st.session_state['conversation_state'] = result
                    messages = result.get("messages", [])
                    
                    # Find the assistant response
                    assistant_response = "I'm here to help with your nutrition planning!"
                    
                    from langchain_core.messages import AIMessage
                    for message in reversed(messages):
                        if isinstance(message, AIMessage):
                            assistant_response = message.content
                            break
                    
                    # Only add to chat history if we got a new response
                    if assistant_response and assistant_response != "I'm here to help with your nutrition planning!":
                        st.session_state['chat_history'].append(("assistant", assistant_response))
                        st.sidebar.success(f"✅ Response received! ({len(assistant_response)} chars)")
                    else:
                        # Fallback response if no AI message found
                        fallback_response = "I processed your message. How can I help you further?"
                        st.session_state['chat_history'].append(("assistant", fallback_response))
                        st.sidebar.warning("⚠️ No AI response found, using fallback")
                    
                    # Auto-save current conversation
                    if username:
                        save_conversation_history(username)
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Agent error: {str(e)}")
                    st.session_state['chat_history'].append(("assistant", f"I encountered an error: {str(e)}. Please try again."))
                    st.sidebar.error(f"❌ Agent error: {str(e)}")
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            st.session_state['chat_history'].append(("assistant", f"I encountered an unexpected error: {str(e)}. Please try again."))
            st.sidebar.error(f"❌ Unexpected error: {str(e)}")
            st.rerun()

def display_quick_actions():
    """Display quick action buttons."""
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🍳 Plan Today's Meals"):
            if st.session_state['agent_graph']:
                # Add this as a user message to trigger the normal chat flow
                quick_message = "Please create a meal plan for today based on my profile and any constraints I mentioned."
                st.session_state['chat_history'].append(("user", quick_message))
                
                # Process through normal chat interface logic
                try:
                    agent_graph = st.session_state['agent_graph']
                    conversation_state = st.session_state['conversation_state'].copy()
                    
                    from langchain_core.messages import HumanMessage
                    current_messages = conversation_state.get("messages", [])
                    
                    # Get context from selected previous chats
                    username = st.session_state.get('username')
                    selected_memory_chats = st.session_state.get('selected_memory_chats', [])
                    
                    enhanced_message = quick_message
                    if username and selected_memory_chats:
                        memory_context = get_selected_chat_context(username, selected_memory_chats)
                        if memory_context:
                            enhanced_message = f"{memory_context}\n\nCURRENT USER MESSAGE: {quick_message}"
                    
                    conversation_state["messages"] = current_messages + [HumanMessage(content=enhanced_message)]
                    
                    config = {"recursion_limit": 20}
                    result = agent_graph.invoke(conversation_state, config)
                    
                    # Update session state and chat history
                    st.session_state['conversation_state'] = result
                    messages = result.get("messages", [])
                    
                    from langchain_core.messages import AIMessage
                    for message in reversed(messages):
                        if isinstance(message, AIMessage):
                            st.session_state['chat_history'].append(("assistant", message.content))
                            break
                    
                    # Auto-save
                    if username:
                        save_conversation_history(username)
                    
                    st.success("🍳 Meal plan generated! Check the chat.")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error generating meal plan: {e}")
    
    with col2:
        if st.button("📊 Calculate Nutrition Targets"):
            if st.session_state['agent_graph']:
                # Add this as a user message to trigger the normal chat flow
                quick_message = "Please calculate my daily nutrition targets based on my profile."
                st.session_state['chat_history'].append(("user", quick_message))
                
                try:
                    agent_graph = st.session_state['agent_graph']
                    conversation_state = st.session_state['conversation_state'].copy()
                    
                    from langchain_core.messages import HumanMessage
                    current_messages = conversation_state.get("messages", [])
                    
                    # Get context from selected previous chats
                    username = st.session_state.get('username')
                    selected_memory_chats = st.session_state.get('selected_memory_chats', [])
                    
                    enhanced_message = quick_message
                    if username and selected_memory_chats:
                        memory_context = get_selected_chat_context(username, selected_memory_chats)
                        if memory_context:
                            enhanced_message = f"{memory_context}\n\nCURRENT USER MESSAGE: {quick_message}"
                    
                    conversation_state["messages"] = current_messages + [HumanMessage(content=enhanced_message)]
                    
                    config = {"recursion_limit": 20}
                    result = agent_graph.invoke(conversation_state, config)
                    
                    # Update session state and chat history
                    st.session_state['conversation_state'] = result
                    messages = result.get("messages", [])
                    
                    from langchain_core.messages import AIMessage
                    for message in reversed(messages):
                        if isinstance(message, AIMessage):
                            st.session_state['chat_history'].append(("assistant", message.content))
                            break
                    
                    # Auto-save
                    if username:
                        save_conversation_history(username)
                    
                    st.success("📊 Nutrition targets calculated! Check the chat.")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error calculating targets: {e}")
    
    with col3:
        if st.button("� Recipe Analyzer"):
            st.session_state['show_recipe_analyzer'] = True
    
    with col4:
        if st.button("📈 Daily Results"):
            st.session_state['show_daily_results'] = True

def display_recipe_analyzer():
    """Display standalone recipe Analyzer with ML analysis."""
    if st.session_state.get('show_recipe_analyzer', False):
        with st.expander("� Recipe Analyzer", expanded=True):
            st.markdown("### Enter Recipe Description")
            recipe_text = st.text_area(
                "Describe your recipe:",
                placeholder="e.g., Grilled chicken breast with quinoa and steamed broccoli\nor\nPasta with tomato sauce and basil\nor\nChocolate chip cookies",
                height=100,
                help="You can enter just a recipe name, or include ingredients and steps. The AI will enhance your description."
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                analyze_button = st.button("🔍 Analyze Recipe", type="primary")
            
            with col2:
                if st.button("❌ Close"):
                    st.session_state['show_recipe_analyzer'] = False
                    st.rerun()
            
            # Perform analysis when button is clicked
            if analyze_button and recipe_text.strip():
                with st.spinner("Analyzing recipe with AI and ML models..."):
                    try:
                        # Initialize model predictor if not available
                        if ('food_predictor' not in st.session_state) or (st.session_state['food_predictor'] is None):
                            from model_predictor import FoodModelPredictor
                            api_key = st.session_state.get('api_key')
                            if not api_key:
                                import os
                                api_key = os.environ.get("GOOGLE_API_KEY", "")
                                if api_key:
                                    st.sidebar.info("Loaded API key from environment.")
                            if api_key:
                                st.session_state['food_predictor'] = FoodModelPredictor(api_key=api_key)
                            else:
                                st.error("Google API key required for recipe analysis")
                                return
                        
                        food_predictor = st.session_state['food_predictor']
                        
                        # Analyze the recipe
                        analysis = food_predictor.analyze_recipe(recipe_text.strip())
                        
                        if "error" in analysis:
                            st.error(f"Analysis failed: {analysis['error']}")
                            return
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("## 📊 Recipe Analysis")
                        
                        # Enhanced Recipe Information
                        import json
                        enhanced_recipe = analysis.get('enhanced_recipe', {})
                        recipe_data = None
                        # If enhanced_recipe is a string, parse it as JSON
                        if isinstance(enhanced_recipe, str):
                            try:
                                recipe_data = json.loads(enhanced_recipe)
                            except Exception:
                                recipe_data = None
                        elif isinstance(enhanced_recipe, dict):
                            recipe_data = enhanced_recipe
                        if recipe_data:
                            recipe_name = recipe_data.get('name', 'N/A')
                            ingredients = recipe_data.get('ingredients', [])
                            steps = recipe_data.get('steps', [])

                            # If ingredients is a string, try to parse as JSON
                            if isinstance(ingredients, str):
                                try:
                                    parsed = json.loads(ingredients)
                                    if isinstance(parsed, list):
                                        ingredients = parsed
                                    elif isinstance(parsed, dict):
                                        ingredients = parsed.get('ingredients', [])
                                except Exception:
                                    ingredients = [ingredients]

                            # If steps is a string, try to parse as JSON
                            if isinstance(steps, str):
                                try:
                                    parsed = json.loads(steps)
                                    if isinstance(parsed, list):
                                        steps = parsed
                                    elif isinstance(parsed, dict):
                                        steps = parsed.get('steps', [])
                                except Exception:
                                    steps = [steps]

                            # Display recipe name
                            st.markdown(f"### 🍽️ {recipe_name}")
                            # Display ingredients
                            if ingredients:
                                st.markdown("#### 🥗 Ingredients")
                                for ingredient in ingredients:
                                    st.markdown(f"• {ingredient}")
                            # Display steps
                            if steps:
                                st.markdown("#### 👨‍🍳 Instructions")
                                for i, step in enumerate(steps, 1):
                                    st.markdown(f"**{i}.** {step}")
                            
                            st.markdown("---")
                        
                        # ML Predictions
                        st.markdown("### 🤖 ML Model Predictions")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            # Difficulty Analysis
                            difficulty = analysis.get('difficulty', {})
                            st.markdown("#### � Difficulty")
                            if 'all_probabilities' in difficulty:
                                for label, prob in difficulty['all_probabilities'].items():
                                    st.write(f"{label}: {prob:.1%}")
                            else:
                                st.write(f"{difficulty.get('prediction', 'Unknown')}: {difficulty.get('confidence', 0):.1%}")
                        with col2:
                            # Meal Type Analysis
                            meal_type = analysis.get('meal_type', {})
                            st.markdown("#### 🍽️ Meal Type")
                            if 'all_probabilities' in meal_type:
                                for label, prob in meal_type['all_probabilities'].items():
                                    st.write(f"{label.title()}: {prob:.1%}")
                            else:
                                st.write(f"{meal_type.get('prediction', 'Unknown')}: {meal_type.get('confidence', 0):.1%}")
                        # ...removed total time display...
                        
                    except Exception as e:
                        st.error(f"Error during recipe analysis: {str(e)}")
                        st.write("Please check your Google API key and try again.")
            
            elif analyze_button and not recipe_text.strip():
                st.warning("Please enter a recipe description to analyze.")

def display_daily_results():
    """Display daily results tracking interface."""
    if st.session_state.get('show_daily_results', False):
        with st.expander("📈 Daily Results Tracking", expanded=True):
            st.write("Track your daily progress and meal compliance:")
            
            # Weight tracking
            col1, col2 = st.columns(2)
            with col1:
                current_weight = st.number_input(
                    "Current weight (kg):",
                    min_value=30.0,
                    max_value=200.0,
                    value=70.0,
                    step=0.1,
                    help="Enter your current weight in kilograms"
                )
            
            with col2:
                date_tracked = st.date_input("Date:")
            
            # Meal compliance tracking
            st.subheader("🍽️ Meal Compliance")
            
            meals = ["Breakfast", "Lunch", "Dinner", "Snacks"]
            meal_data = {}
            
            for meal in meals:
                st.write(f"**{meal}:**")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    followed_exactly = st.checkbox(
                        f"Followed {meal.lower()} plan exactly",
                        key=f"exact_{meal.lower()}"
                    )
                
                with col2:
                    if not followed_exactly:
                        differences = st.text_area(
                            f"Describe differences for {meal.lower()}:",
                            placeholder=f"e.g., Had pasta instead of quinoa, added extra olive oil, skipped the salad...",
                            key=f"diff_{meal.lower()}",
                            height=60
                        )
                    else:
                        differences = "Followed plan exactly"
                
                meal_data[meal] = {
                    "followed_exactly": followed_exactly,
                    "differences": differences if not followed_exactly else "Followed plan exactly"
                }
                
                st.markdown("---")
            
            # Additional information
            additional_info = st.text_area(
                "Additional information:",
                placeholder="Any other relevant information about your day (energy levels, cravings, exercise, water intake, sleep quality, etc.)",
                height=100
            )
            
            # Submit button
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Submit Daily Results") and st.session_state['agent_graph']:
                    # Construct the detailed prompt
                    meal_summary = []
                    for meal, data in meal_data.items():
                        if data["followed_exactly"]:
                            meal_summary.append(f"- {meal}: ✅ Followed plan exactly")
                        else:
                            meal_summary.append(f"- {meal}: ❌ {data['differences']}")
                    
                    meal_text = "\n".join(meal_summary)
                    
                    quick_message = f"""Please analyze my daily results:

📊 **Daily Tracking Results for {date_tracked}**

**Weight:** {current_weight} kg

**Meal Compliance:**
{meal_text}

**Additional Information:** {additional_info if additional_info else "None provided"}

Please provide analysis of my progress, adherence to the meal plan, and any recommendations for improvements. Consider how the weight change and meal compliance align with my nutrition goals."""
                    
                    st.session_state['chat_history'].append(("user", quick_message))
                    
                    try:
                        agent_graph = st.session_state['agent_graph']
                        conversation_state = st.session_state['conversation_state'].copy()
                        
                        from langchain_core.messages import HumanMessage
                        current_messages = conversation_state.get("messages", [])
                        
                        # Get context from selected previous chats
                        username = st.session_state.get('username')
                        selected_memory_chats = st.session_state.get('selected_memory_chats', [])
                        
                        enhanced_message = quick_message
                        if username and selected_memory_chats:
                            memory_context = get_selected_chat_context(username, selected_memory_chats)
                            if memory_context:
                                enhanced_message = f"{memory_context}\n\nCURRENT USER MESSAGE: {quick_message}"
                        
                        conversation_state["messages"] = current_messages + [HumanMessage(content=enhanced_message)]
                        
                        config = {"recursion_limit": 20}
                        result = agent_graph.invoke(conversation_state, config)
                        
                        # Update session state and chat history
                        st.session_state['conversation_state'] = result
                        messages = result.get("messages", [])
                        
                        from langchain_core.messages import AIMessage
                        for message in reversed(messages):
                            if isinstance(message, AIMessage):
                                st.session_state['chat_history'].append(("assistant", message.content))
                                break
                        
                        # Auto-save
                        if username:
                            save_conversation_history(username)
                        
                        st.success("📊 Daily results submitted! Check the chat for analysis.")
                        st.session_state['show_daily_results'] = False
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error submitting daily results: {e}")
            
            with col2:
                if st.button("Close"):
                    st.session_state['show_daily_results'] = False
                    st.rerun()


def inject_user_profile_to_agent():
    """Inject registration data into the agent's conversation state."""
    username = st.session_state.get('username')
    if username and not st.session_state['conversation_state'].get("profile_setup_complete", False):
        user_info = load_user_info(username)
        if user_info:
            # Create a comprehensive profile for the AI agent
            agent_profile = {
                "registration_complete": True,
                "basic_info": {
                    "age": user_info.get('age'),
                    "weight": user_info.get('weight'),
                    "height": user_info.get('height'),
                    "gender": user_info.get('gender'),
                    "bmi": user_info['weight'] / ((user_info['height']/100) ** 2) if user_info.get('weight') and user_info.get('height') else None
                },
                "goals_and_preferences": {
                    "primary_goal": user_info.get('primary_goal'),
                    "activity_level": user_info.get('activity_level'),
                    "dietary_preferences": user_info.get('dietary_preferences', []),
                    "cooking_experience": user_info.get('cooking_experience'),
                    "budget_range": user_info.get('budget_range')
                },
                "lifestyle": {
                    "meal_schedule": user_info.get('meal_schedule', {}),
                    "water_intake_goal": user_info.get('water_intake_goal')
                },
                "health": {
                    "health_conditions": user_info.get('health_conditions', []),
                },
                "food_preferences": {
                    "favorite_cuisines": user_info.get('favorite_cuisines', []),
                    "foods_to_avoid": user_info.get('foods_to_avoid', ''),
                    "additional_notes": user_info.get('additional_notes', '')
                },
                "conversation_context": f"User {username} has completed detailed registration with comprehensive health and dietary information.",
                "llm_understanding": f"This user is a {user_info.get('age')}-year-old {user_info.get('gender', '').lower()} with goal: {user_info.get('primary_goal')}. Activity level: {user_info.get('activity_level')}. Key dietary needs: {', '.join(user_info.get('dietary_preferences', [])[:3])}."
            }
            
            # Inject into conversation state
            st.session_state['conversation_state']["user_profile"] = agent_profile
            st.session_state['conversation_state']["profile_setup_complete"] = True
            
            # Add initial context message for the AI
            from langchain_core.messages import HumanMessage
            
            # Create meal schedule summary for the AI
            meal_schedule = user_info.get('meal_schedule', {})
            meal_summary = []
            for meal_name, meal_info in meal_schedule.items():
                if meal_name != 'snacks' and meal_info.get('enabled'):
                    meal_summary.append(f"{meal_name}: {meal_info.get('location')} with {meal_info.get('cooking_time')} cooking time")
                elif meal_name == 'snacks' and meal_info.get('enabled'):
                    meal_summary.append(f"snacks: {meal_info.get('frequency')}, {meal_info.get('type')}")
            
            initial_context = f"""Hi! I've just completed my registration with detailed information about my health, goals, and preferences. 
            My profile includes: {user_info.get('age')} years old, {user_info.get('gender')}, {user_info.get('weight')}kg, {user_info.get('height')}cm.
            My primary goal is to {user_info.get('primary_goal', '').lower()}. 
            My activity level: {user_info.get('activity_level')}.
            Dietary preferences/restrictions: {', '.join(user_info.get('dietary_preferences', []))}.
            
            My meal schedule:
            {chr(10).join(f"• {meal}" for meal in meal_summary) if meal_summary else "• Standard 3 meals per day"}
            
            Please use this information to provide personalized nutrition advice and meal planning!"""
            
            current_messages = st.session_state['conversation_state'].get("messages", [])
            if not any("completed my registration" in str(msg) for msg in current_messages):
                st.session_state['conversation_state']["messages"] = current_messages + [HumanMessage(content=initial_context)]

def main_app():
    # Initialize session state
    initialize_session_state()
    
    # Ensure we have a chat session ID
    if not st.session_state.get('chat_session_id'):
        start_new_chat_session()
    
    # Inject user registration data into agent conversation state
    inject_user_profile_to_agent()
    
    # Header
    st.markdown('<h1 class="main-header">🥗 AI Nutritionist Agent</h1>', unsafe_allow_html=True)
    st.markdown("---")
    # Sidebar
    with st.sidebar:
        st.title("🛠️ Control Panel")
        # Initialize agent
        if not init_agent():
            st.stop()
        st.markdown("---")
        display_user_profile()
        
        # Show conversation history status
        username = st.session_state.get('username')
        chat_count = len(st.session_state.get('chat_history', []))
        current_session = st.session_state.get('chat_session_id')
        
        if username and current_session:
            st.sidebar.info(f"💬 Chat Session: {current_session}")
            if chat_count > 0:
                st.sidebar.info(f"� {chat_count} messages in current chat")
            
            # Show previous chat sessions
            sessions = list_user_chat_sessions(username)
            if sessions:
                with st.sidebar.expander(f"📚 Previous Chats ({len(sessions)})", expanded=False):
                    for session in sessions[:5]:  # Show last 5 sessions
                        if session['session_id'] != current_session:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**{session['session_id']}**")
                                st.write(f"{session['message_count']} messages")
                            with col2:
                                if st.button("📂", key=f"load_{session['session_id']}", help="Load this chat"):
                                    # Save current chat first
                                    if st.session_state.get('chat_history'):
                                        save_conversation_history(username)
                                    
                                    # Load the selected chat session
                                    if load_conversation_history(username, session['session_id']):
                                        st.success(f"Loaded chat session: {session['session_id']}")
                                        st.rerun()
                            st.markdown("---")
            
            # Memory selection for current chat
            available_sessions = [s for s in sessions if s['session_id'] != current_session] if sessions else []
            
            if available_sessions:
                with st.sidebar.expander("🧠 Memory Settings", expanded=False):
                    st.write("Select previous chats to include as context:")
                    
                    selected_memory = st.session_state.get('selected_memory_chats', [])
                    
                    for session in available_sessions[:6]:  # Show up to 6 previous chats
                        session_id = session['session_id']
                        preview = session['preview'] if session['preview'] else "No preview"
                        
                        is_selected = session_id in selected_memory
                        
                        # Use unique key for each checkbox
                        if st.checkbox(f"**{session_id}** ({session['message_count']} msgs)", 
                                     value=is_selected, 
                                     key=f"memory_select_{session_id}",
                                     help=f"Preview: {preview[:50]}..."):
                            if session_id not in selected_memory:
                                selected_memory.append(session_id)
                        else:
                            if session_id in selected_memory:
                                selected_memory.remove(session_id)
                    
                    st.session_state['selected_memory_chats'] = selected_memory
                    
                    if selected_memory:
                        st.info(f"🧠 {len(selected_memory)} chat(s) selected as memory")
                    else:
                        st.info("💭 No memory context selected")
        
        st.markdown("---")
        st.subheader("⚙️ Settings")
        
        if st.button("🗑️ Clear Current Chat"):
            # Clear only current session memory, keep chat session ID
            st.session_state['conversation_state'] = {
                "messages": [],
                "user_profile": {},
                "weight_goal": {},
                "daily_constraints": {},
                "previous_results": {},
                "meal_plan": {},
                "nutritional_values": {},
                "profile_setup_complete": False
            }
            st.session_state['chat_history'] = []
            
            # Re-inject user profile for this session
            inject_user_profile_to_agent()
            
            st.success(f"💫 Current chat cleared! (Session: {st.session_state.get('chat_session_id', 'unknown')})")
            st.rerun()
            
        if st.button("🆕 New Conversation"):
            # Save current conversation to disk first
            username = st.session_state.get('username')
            if username and st.session_state.get('chat_history'):
                save_conversation_history(username)
                st.success(f"💾 Previous chat saved! (Session: {st.session_state.get('chat_session_id', 'unknown')})")
            
            # Start completely new chat session
            start_new_chat_session()
            st.success(f"🆕 New conversation started! (Session: {st.session_state.get('chat_session_id', 'unknown')})")
            st.rerun()
            
        if st.button("🔧 Test Agent"):
            if st.session_state['agent_graph']:
                try:
                    from langchain_core.messages import HumanMessage
                    test_state = {"messages": [HumanMessage(content="Hello, this is a test.")]}
                    result = st.session_state['agent_graph'].invoke(test_state, {"recursion_limit": 5})
                    st.success("✅ Agent test successful!")
                    if result and result.get("messages"):
                        st.write(f"Response length: {len(str(result.get('messages', [])[-1]))} characters")
                except Exception as e:
                    st.error(f"❌ Agent test failed: {str(e)}")
            else:
                st.error("❌ Agent not initialized")
                
        if st.button("📊 Debug Info"):
            st.write("**Session State Info:**")
            st.write(f"Agent: {'✅' if st.session_state['agent_graph'] else '❌'}")
            st.write(f"Profile: {'✅' if st.session_state['conversation_state'].get('profile_setup_complete') else '❌'}")
            st.write(f"Messages: {len(st.session_state['conversation_state'].get('messages', []))}")
            st.write(f"Chat history: {len(st.session_state['chat_history'])}")
            
            # Show conversation history status
            username = st.session_state.get('username')
            if username:
                conversation_file = get_conversation_file(username)
                if conversation_file.exists():
                    try:
                        file_size = conversation_file.stat().st_size
                        st.write(f"💾 Conversation file: {file_size} bytes")
                        
                        # Show last update time
                        with open(conversation_file, "r") as f:
                            data = json.load(f)
                            last_updated = data.get('last_updated', 'Unknown')
                            st.write(f"🕒 Last saved: {last_updated}")
                    except:
                        st.write("💾 Conversation file: Error reading")
                else:
                    st.write("💾 Conversation file: Not found")
                
                # Show archived conversations
                user_dir = Path(__file__).parent / "secrets"
                archive_files = list(user_dir.glob(f"{username}_conversation_*.json"))
                if archive_files:
                    st.write(f"📁 Archived conversations: {len(archive_files)}")
                    for archive_file in sorted(archive_files)[-3:]:  # Show last 3
                        try:
                            with open(archive_file, "r") as f:
                                archive_data = json.load(f)
                                msg_count = archive_data.get('message_count', 0)
                                archived_at = archive_data.get('archived_at', 'Unknown')
                                st.write(f"  • {archive_file.name}: {msg_count} messages")
                        except:
                            st.write(f"  • {archive_file.name}: Error reading")
            
        if st.button("📥 Export Conversation"):
            conversation_data = {
                "conversation_state": st.session_state['conversation_state'],
                "chat_history": st.session_state['chat_history'],
                "export_date": datetime.now().isoformat()
            }
            st.download_button(
                "Download JSON",
                data=json.dumps(conversation_data, indent=2),
                file_name=f"nutrition_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🍽️ Meal Plan", "📊 Dashboard", "⚡ Quick Actions"])
    with tab1:
        display_chat_interface()
    with tab2:
        display_meal_plan()
    with tab3:
        display_nutrition_dashboard()
    with tab4:
        display_quick_actions()
        display_recipe_analyzer()
        display_daily_results()
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
            AI Nutritionist Agent powered by LangGraph, Google Gemini, and Custom ML Models
        </div>
        """, 
        unsafe_allow_html=True
    )

# --- Main Application Logic ---

# --- Login and Registration Flow ---

def get_users_file():
    # Use nut_agent/secrets directory for users.json
    return Path(__file__).parent / "secrets" / "users.json"

def get_conversation_file(username, session_id=None):
    # Use nut_agent/secrets directory for user conversation history
    if session_id:
        return Path(__file__).parent / "secrets" / f"{username}_chat_{session_id}.json"
    else:
        # Current active session
        current_session = st.session_state.get('chat_session_id')
        if current_session:
            return Path(__file__).parent / "secrets" / f"{username}_chat_{current_session}.json"
        else:
            # Fallback for backwards compatibility
            return Path(__file__).parent / "secrets" / f"{username}_conversations.json"

def generate_new_chat_session_id():
    """Generate a new unique chat session ID."""
    import uuid
    return str(uuid.uuid4())[:8]  # Short 8-character ID

def start_new_chat_session():
    """Start a new chat session with a fresh ID."""
    st.session_state['chat_session_id'] = generate_new_chat_session_id()
    st.session_state['chat_history'] = []
    st.session_state['conversation_state'] = {
        "messages": [],
        "user_profile": {},
        "weight_goal": {},
        "daily_constraints": {},
        "previous_results": {},
        "meal_plan": {},
        "nutritional_values": {},
        "profile_setup_complete": False
    }
    
    # Load user profile from users.json into this chat session
    username = st.session_state.get('username')
    if username:
        inject_user_profile_to_agent()

def save_conversation_history(username, session_id=None):
    """Save the current conversation history and state to disk."""
    if not username:
        return
    
    conversation_file = get_conversation_file(username, session_id)
    
    try:
        # Convert conversation state messages to serializable format
        conversation_state = st.session_state.get('conversation_state', {}).copy()
        
        # Convert LangChain messages to serializable format
        if 'messages' in conversation_state:
            serializable_messages = []
            for msg in conversation_state['messages']:
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    # LangChain message object
                    serializable_messages.append({
                        'type': msg.type,
                        'content': msg.content
                    })
                else:
                    # Already serializable
                    serializable_messages.append(msg)
            conversation_state['messages'] = serializable_messages
        
        conversation_data = {
            "chat_session_id": st.session_state.get('chat_session_id'),
            "chat_history": st.session_state.get('chat_history', []),
            "conversation_state": conversation_state,
            "last_updated": datetime.now().isoformat(),
            "version": "2.0"  # Updated version for session-based chats
        }
        
        with open(conversation_file, "w") as f:
            json.dump(conversation_data, f, indent=2)
            f.flush()
            
        print(f"DEBUG: Saved conversation history for {username} (session: {st.session_state.get('chat_session_id')})")
        
    except Exception as e:
        print(f"ERROR saving conversation history: {e}")
        st.error(f"Could not save conversation history: {e}")

def load_conversation_history(username, session_id=None):
    """Load conversation history and state from disk."""
    if not username:
        return False
    
    conversation_file = get_conversation_file(username, session_id)
    
    try:
        if conversation_file.exists() and conversation_file.stat().st_size > 0:
            with open(conversation_file, "r") as f:
                content = f.read().strip()
                if content:
                    conversation_data = json.loads(content)
                    
                    # Load chat session ID
                    if 'chat_session_id' in conversation_data:
                        st.session_state['chat_session_id'] = conversation_data['chat_session_id']
                    
                    # Load chat history
                    st.session_state['chat_history'] = conversation_data.get('chat_history', [])
                    
                    # Load conversation state
                    loaded_state = conversation_data.get('conversation_state', {})
                    if loaded_state:
                        # Convert serialized messages back to LangChain format if needed
                        if 'messages' in loaded_state:
                            from langchain_core.messages import HumanMessage, AIMessage
                            langchain_messages = []
                            for msg in loaded_state['messages']:
                                if isinstance(msg, dict) and 'type' in msg and 'content' in msg:
                                    # Convert back to LangChain message
                                    if msg['type'] == 'human':
                                        langchain_messages.append(HumanMessage(content=msg['content']))
                                    elif msg['type'] == 'ai':
                                        langchain_messages.append(AIMessage(content=msg['content']))
                                else:
                                    # Keep as is if already in correct format
                                    langchain_messages.append(msg)
                            loaded_state['messages'] = langchain_messages
                        
                        st.session_state['conversation_state'] = loaded_state
                    
                    print(f"DEBUG: Loaded conversation history for {username} (session: {st.session_state.get('chat_session_id')})")
                    print(f"DEBUG: Loaded {len(st.session_state['chat_history'])} chat messages")
                    return True
        
        return False
        
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not parse conversation data: {e}")
        return False
    except Exception as e:
        print(f"ERROR loading conversation history: {e}")
        return False

def clear_conversation_history(username, session_id=None):
    """Clear conversation history from disk."""
    if not username:
        return
    
    conversation_file = get_conversation_file(username, session_id)
    
    try:
        if conversation_file.exists():
            conversation_file.unlink()
            print(f"DEBUG: Cleared conversation history file for {username} (session: {session_id or st.session_state.get('chat_session_id')})")
    except Exception as e:
        print(f"ERROR clearing conversation history: {e}")

def get_selected_chat_context(username, selected_session_ids, max_messages_per_chat=10):
    """Get context from selected previous chats to include as memory."""
    if not username or not selected_session_ids:
        return ""
    
    context_parts = []
    
    for session_id in selected_session_ids:
        try:
            chat_file = get_conversation_file(username, session_id)
            if chat_file.exists():
                with open(chat_file, "r") as f:
                    data = json.load(f)
                    chat_history = data.get('chat_history', [])
                    
                    if chat_history:
                        # Get last N messages from this chat
                        recent_messages = chat_history[-max_messages_per_chat:]
                        
                        session_context = f"\n--- Previous Chat Session {session_id} ---\n"
                        for role, message in recent_messages:
                            if role == "user":
                                session_context += f"User: {message}\n"
                            else:
                                session_context += f"AI: {message}\n"
                        session_context += "--- End Previous Chat ---\n"
                        
                        context_parts.append(session_context)
        except Exception as e:
            print(f"Error loading context from session {session_id}: {e}")
            continue
    
    if context_parts:
        full_context = "\n".join(context_parts)
        return f"""
CONTEXT FROM PREVIOUS CONVERSATIONS:
{full_context}

Please use this context from previous conversations to provide continuity and build upon past discussions when relevant to the current user message.
"""
    return ""

def list_user_chat_sessions(username):
    """List all chat sessions for a user."""
    if not username:
        return []
    
    user_dir = Path(__file__).parent / "secrets"
    chat_files = list(user_dir.glob(f"{username}_chat_*.json"))
    sessions = []
    
    for chat_file in chat_files:
        try:
            with open(chat_file, "r") as f:
                data = json.load(f)
                # Get a preview of the conversation
                chat_history = data.get('chat_history', [])
                preview = ""
                if chat_history:
                    # Get first user message as preview
                    for role, message in chat_history:
                        if role == "user" and len(message) > 10:
                            preview = message[:60] + "..." if len(message) > 60 else message
                            break
                
                sessions.append({
                    "session_id": data.get('chat_session_id', 'unknown'),
                    "file_name": chat_file.name,
                    "message_count": len(chat_history),
                    "last_updated": data.get('last_updated', 'unknown'),
                    "preview": preview
                })
        except:
            continue
    
    return sorted(sessions, key=lambda x: x['last_updated'], reverse=True)

def save_user_info(username, user_info):
    st.session_state['user_info'][username] = user_info
    users_file = get_users_file()
    try:
        # Always ensure we have a dictionary to work with
        all_users = {}
        if users_file.exists() and users_file.stat().st_size > 0:
            try:
                with open(users_file, "r") as f:
                    content = f.read().strip()
                    if content:  # Only parse if there's actual content
                        all_users = json.loads(content)
            except json.JSONDecodeError:
                st.warning("users.json was corrupted, creating new file")
                all_users = {}
        
        all_users[username] = user_info
        
        # Write with proper formatting
        with open(users_file, "w") as f:
            json.dump(all_users, f, indent=2)
            f.flush()
        
        st.success(f"✅ User {username} saved successfully!")
        print(f"DEBUG: Saved user info for {username} to {users_file}")
        
    except Exception as e:
        st.error(f"Could not save user info: {e}")
        print(f"ERROR saving user info: {e}")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_user_info(username):
    # Try session first
    if username in st.session_state['user_info']:
        return st.session_state['user_info'][username]
    
    # Try disk
    users_file = get_users_file()
    try:
        if users_file.exists() and users_file.stat().st_size > 0:
            with open(users_file, "r") as f:
                content = f.read().strip()
                if content:  # Only parse if there's actual content
                    all_users = json.loads(content)
                    return all_users.get(username, None)
        
        # File doesn't exist or is empty
        return None
        
    except json.JSONDecodeError as e:
        st.warning(f"Could not parse user data: {e}")
        return None
    except Exception as e:
        st.warning(f"Could not load user info: {e}")
        return None

def landing_page():
    st.title("Welcome to AI Nutritionist Agent")
    username = st.text_input("Username", key="landing_username")
    password = st.text_input("Password", type="password", key="landing_password")
    col1, col2 = st.columns(2)
    with col1:
        login_btn = st.button("Login")
    with col2:
        register_btn = st.button("Register")
    if login_btn and username and password:
        user_info = load_user_info(username)
        if user_info and 'password_hash' in user_info:
            if user_info['password_hash'] == hash_password(password):
                st.session_state['username'] = username
                st.session_state['password'] = password
                st.session_state['is_new_user'] = False
                st.session_state['initialized'] = True
                
                # Start a new chat session (each login = new chat)
                start_new_chat_session()
                
                # Show available chat sessions
                sessions = list_user_chat_sessions(username)
                if sessions:
                    st.success(f"Login successful! You have {len(sessions)} previous chat sessions. Started new chat session: {st.session_state['chat_session_id']}")
                else:
                    st.success(f"Login successful! Started new chat session: {st.session_state['chat_session_id']}")
                
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
        elif user_info:
            st.error("This account was created without a password. Please register again.")
        else:
            st.error("User not found. Please register.")
    elif register_btn and username and password:
        st.session_state['username'] = username
        st.session_state['password'] = password
        st.session_state['show_register'] = True
        st.rerun()

def registration_form():
    st.title("Register New Account")
    username = st.session_state.get('username', '')
    password = st.session_state.get('password', '')
    st.write(f"Username: **{username}**")
    st.write(f"Password: **{'*' * len(password)}**")
    
    st.markdown("---")
    st.subheader("📊 Basic Information")
    
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        age = st.number_input("Age", min_value=5, max_value=120, value=30)
    
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        activity_level = st.selectbox(
            "Activity Level", 
            ["Sedentary (desk job, no exercise)", 
             "Lightly active (light exercise 1-3 days/week)",
             "Moderately active (moderate exercise 3-5 days/week)", 
             "Very active (hard exercise 6-7 days/week)",
             "Extremely active (very hard exercise, physical job)"]
        )
    
    st.markdown("---")
    st.subheader("🎯 Goals & Preferences")
    
    col3, col4 = st.columns(2)
    with col3:
        primary_goal = st.selectbox(
            "Primary Goal",
            ["Lose weight", "Gain weight", "Maintain weight", "Build muscle", 
             "Improve health", "Increase energy", "Better sleep"]
        )
        
        dietary_preferences = st.multiselect(
            "Dietary Preferences/Restrictions",
            ["None", "Vegetarian", "Vegan", "Pescatarian", "Keto", "Paleo", 
             "Low-carb", "Low-fat", "Gluten-free", "Dairy-free", "Halal", 
             "Kosher", "Nut allergies", "Shellfish allergies"]
        )
    
    with col4:
        cooking_experience = st.selectbox(
            "Cooking Experience",
            ["Beginner (prefer simple recipes)", 
             "Intermediate (comfortable with most recipes)",
             "Advanced (enjoy complex cooking)"]
        )
        
        budget_range = st.selectbox(
            "Weekly Food Budget",
            ["$50-100", "$100-150", "$150-200", "$200-300", "$300+", "No specific budget"]
        )
    
    st.markdown("---")
    st.subheader("🍽️ Meal Planning & Lifestyle")
    
    # Meal preferences section
    st.markdown("**🕐 Meal Schedule & Cooking Time**")
    col5a, col5b = st.columns(2)
    
    with col5a:
        # Breakfast settings
        st.markdown("**🌅 Breakfast**")
        breakfast_enabled = st.checkbox("I eat breakfast", value=True)
        breakfast_location = st.selectbox(
            "Breakfast location",
            ["At home", "On the go", "At work/office", "Varies"],
            key="breakfast_location"
        )
        breakfast_time = st.selectbox(
            "Breakfast cooking time",
            ["0 minutes (buy/order food)", "5-10 minutes", "15-20 minutes", "30+ minutes"],
            key="breakfast_time"
        )
        
        # Lunch settings
        st.markdown("**🌞 Lunch**")
        lunch_enabled = st.checkbox("I eat lunch", value=True)
        lunch_location = st.selectbox(
            "Lunch location",
            ["At work/office", "At home", "Restaurant/cafeteria", "Varies"],
            key="lunch_location"
        )
        lunch_time = st.selectbox(
            "Lunch cooking time",
            ["0 minutes (buy/order food)", "5-10 minutes", "15-20 minutes", "30+ minutes", "Meal prep (weekend prep)"],
            key="lunch_time"
        )
    
    with col5b:
        # Dinner settings
        st.markdown("**🌙 Dinner**")
        dinner_enabled = st.checkbox("I eat dinner", value=True)
        dinner_location = st.selectbox(
            "Dinner location",
            ["At home", "Restaurant", "At work (late shift)", "Varies"],
            key="dinner_location"
        )
        dinner_time = st.selectbox(
            "Dinner cooking time",
            ["0 minutes (buy/order food)", "15-30 minutes", "30-60 minutes", "1+ hours", "Varies by day"],
            key="dinner_time"
        )
        
        # Snacks settings
        st.markdown("**🍎 Snacks**")
        snacks_enabled = st.checkbox("I eat snacks", value=True)
        snacks_frequency = st.selectbox(
            "Snack frequency",
            ["No snacks", "1 snack per day", "2 snacks per day", "3+ snacks per day", "As needed"],
            key="snacks_frequency"
        )
        snacks_type = st.selectbox(
            "Preferred snack type",
            ["Quick/ready-made", "Light prep (5 min)", "Homemade", "Mixed", "Healthy options only"],
            key="snacks_type"
        )
    
    st.markdown("---")
    st.subheader("🏥 Health & Constraints")
    
    col6, col7 = st.columns(2)
    with col7:
        health_conditions = st.multiselect(
            "Health Conditions (optional)",
            ["None", "Diabetes", "High blood pressure", "High cholesterol", 
             "Heart disease", "PCOS", "Thyroid issues", "IBS/Digestive issues", 
             "Food allergies", "Other"]
        )
        
        water_intake_goal = st.selectbox(
            "Daily Water Intake Goal",
            ["1-2 liters", "2-3 liters", "3-4 liters", "4+ liters", "Not sure"]
        )
    
    st.markdown("---")
    st.subheader("🌍 Food Preferences")
    
    favorite_cuisines = st.multiselect(
        "Favorite Cuisines",
        ["American", "Italian", "Mexican", "Asian (Chinese/Thai/Japanese)", 
         "Indian", "Mediterranean", "Middle Eastern", "French", "German", 
         "African", "Latin American", "No preference"]
    )
    
    foods_to_avoid = st.text_area(
        "Foods you dislike or want to avoid (optional)",
        placeholder="e.g., mushrooms, spicy food, raw fish...",
        height=60
    )
    
    additional_notes = st.text_area(
        "Additional notes or specific requirements (optional)",
        placeholder="e.g., work schedule, family considerations, specific health goals...",
        height=80
    )
    
    st.markdown("---")
    submit_btn = st.button("🚀 Complete Registration", type="primary")
    
    if submit_btn:
        if not username:
            st.error("Username is required.")
        elif not password:
            st.error("Password is required.")
        else:
            user_info = {
                'weight': weight,
                'height': height,
                'age': age,
                'gender': gender,
                'activity_level': activity_level,
                'primary_goal': primary_goal,
                'dietary_preferences': dietary_preferences,
                'cooking_experience': cooking_experience,
                'budget_range': budget_range,
                'meal_schedule': {
                    'breakfast': {
                        'enabled': breakfast_enabled,
                        'location': breakfast_location,
                        'cooking_time': breakfast_time
                    },
                    'lunch': {
                        'enabled': lunch_enabled,
                        'location': lunch_location,
                        'cooking_time': lunch_time
                    },
                    'dinner': {
                        'enabled': dinner_enabled,
                        'location': dinner_location,
                        'cooking_time': dinner_time
                    },
                    'snacks': {
                        'enabled': snacks_enabled,
                        'frequency': snacks_frequency,
                        'type': snacks_type
                    }
                },
                'health_conditions': health_conditions,
                'water_intake_goal': water_intake_goal,
                'favorite_cuisines': favorite_cuisines,
                'foods_to_avoid': foods_to_avoid,
                'additional_notes': additional_notes,
                'password_hash': hash_password(password),
                'registration_date': datetime.now().isoformat()
            }
            save_user_info(username, user_info)
            st.session_state['is_new_user'] = False
            st.session_state['initialized'] = True
            st.session_state['show_register'] = False
            st.session_state['show_login'] = False
            st.success("🎉 Registration complete! Welcome to your AI Nutritionist!")
            st.info("💡 Your detailed profile will be automatically shared with the AI to provide personalized recommendations.")
            
            # Show what data will be used
            with st.expander("📋 Your Profile Summary", expanded=True):
                st.write(f"**Basic Info:** {age} years old, {gender}, {weight}kg, {height}cm")
                st.write(f"**Goal:** {primary_goal}")
                st.write(f"**Activity:** {activity_level}")
                st.write(f"**Diet:** {', '.join(dietary_preferences) if dietary_preferences else 'No restrictions'}")
                
                # Display meal schedule
                st.markdown("**🍽️ Meal Schedule:**")
                meal_summary = []
                if breakfast_enabled:
                    meal_summary.append(f"• Breakfast ({breakfast_location}, {breakfast_time})")
                if lunch_enabled:
                    meal_summary.append(f"• Lunch ({lunch_location}, {lunch_time})")
                if dinner_enabled:
                    meal_summary.append(f"• Dinner ({dinner_location}, {dinner_time})")
                if snacks_enabled:
                    meal_summary.append(f"• Snacks ({snacks_frequency}, {snacks_type})")
                
                for meal in meal_summary:
                    st.write(meal)
                
                st.write(f"**Health Conditions:** {', '.join(health_conditions) if health_conditions else 'None specified'}")
                if favorite_cuisines:
                    st.write(f"**Favorite Cuisines:** {', '.join(favorite_cuisines)}")
                if foods_to_avoid:
                    st.write(f"**Foods to Avoid:** {foods_to_avoid}")
            
            st.info("🤖 **Next Steps:** Start chatting with your AI Nutritionist! It already knows your preferences and will provide personalized meal plans and nutrition advice.")
            st.rerun()

def run():
    initialize_session_state()
    if not st.session_state.get('initialized', False):
        if st.session_state['show_register']:
            registration_form()
        else:
            landing_page()
    else:
        main_app()

if __name__ == "__main__":
    run()
