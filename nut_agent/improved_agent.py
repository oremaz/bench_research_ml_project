"""
Improved version of your original LangGraph nutritionist agent.
This version integrates with the ML models and uses better state management.
"""

import os
from typing import Annotated, List, Literal, Dict, Any, Union
from typing_extensions import TypedDict

# Import LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Import LangChain components for Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.tools import tool

# Import our components
from model_predictor import FoodModelPredictor
from utils import (
    calculate_bmi,
    validate_nutrition_targets
)
from config import (
    GEMINI_MODEL, 
    MAX_RECURSION_LIMIT, 
    BMR_CONSTANTS, 
    ACTIVITY_MULTIPLIERS,
    WEIGHT_GOAL_ADJUSTMENTS,
    MACRO_RATIOS,
    WATER_ML_PER_KG
)

class NutritionistState(TypedDict):
    """State representing the nutritionist agent conversation."""
    
    # The chat conversation history
    messages: Annotated[list, add_messages]
    
    # User profile information
    user_profile: Dict[str, Any]
    
    # User's weight goal
    weight_goal: Dict[str, Any]
    
    # Daily constraints
    daily_constraints: Dict[str, Any]
    
    # Previous day's results
    previous_results: Dict[str, Any]
    
    # Generated meal plan
    meal_plan: Dict[str, Any]
    
    # Calculated nutritional values
    nutritional_values: Dict[str, Any]
    
    # Flag indicating if initial profile setup is complete
    profile_setup_complete: bool

# Initialize the LLM and food predictor globally
llm = None
food_predictor = None

def initialize_components(google_api_key: str):
    """Initialize global components."""
    global llm, food_predictor
    
    os.environ["GOOGLE_API_KEY"] = google_api_key
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
    food_predictor = FoodModelPredictor(api_key=google_api_key)

@tool
def validate_recipe_with_ml_models(recipe_description: str, llm_analysis: str = None) -> Dict[str, Any]:
    """
    OPTIONAL: Validate and double-check LLM analysis using trained ML models.
    This tool provides additional validation rather than primary analysis.
    
    Args:
        recipe_description: Description of the recipe
        llm_analysis: Optional LLM analysis to compare against
        
    Returns:
        Dictionary containing ML validation of the recipe
    """
    global food_predictor
    
    if food_predictor is None:
        return {"error": "ML validation not available - food predictor not initialized"}
    
    try:
        # Use ML models to analyze the recipe with text embeddings
        analysis = food_predictor.analyze_recipe(recipe_description)
        
        # Check if analysis was successful
        if "error" in analysis:
            return {
                "error": f"ML analysis failed: {analysis['error']}",
                "recipe_description": recipe_description
            }
        
        # Generate LLM interpretation of the results
        llm_interpretation = food_predictor.generate_llm_interpretation(analysis)
        
        validation_result = {
            "recipe_description": recipe_description,
            "enhanced_recipe": analysis.get("enhanced_recipe", {}),
            "ml_analysis": {
                "difficulty": analysis.get("difficulty", {}),
                "meal_type": analysis.get("meal_type", {}),
                "predicted_nutrients": analysis.get("nutrients", {}),
                "llm_interpretation": llm_interpretation
            },
            "technical_details": {
                "formatted_text": analysis.get("formatted_text", ""),
                "embedding_info": analysis.get("embeddings", {}),
                "models_used": "Text embedding + trained ML models"
            },
            "validation_successful": True
        }
        
        # If LLM analysis provided, add comparison
        if llm_analysis:
            validation_result["llm_comparison"] = {
                "llm_analysis": llm_analysis,
                "note": "Compare ML predictions with LLM analysis for consistency"
            }
        
        return validation_result
        
    except Exception as e:
        return {"error": f"ML validation failed: {str(e)}", "validation_successful": False}

@tool
def calculate_personalized_nutrition_targets(
    weight_kg: float, 
    height_cm: float, 
    age: int, 
    gender: str, 
    activity_level: str,
    weight_goal: str
) -> Dict[str, Any]:
    """
    Calculate personalized daily nutrition targets based on user profile and goals.
    
    Args:
        weight_kg: Current weight in kg
        height_cm: Height in cm
        age: Age in years
        gender: 'male' or 'female'
        activity_level: 'sedentary', 'light', 'moderate', 'active', 'very_active'
        weight_goal: 'lose', 'maintain', 'gain'
        
    Returns:
        Dictionary with personalized daily nutrition targets
    """
    try:
        # Calculate BMR using Mifflin-St Jeor equation
        bmr_const = BMR_CONSTANTS.get(gender.lower(), BMR_CONSTANTS['male'])
        bmr = (bmr_const['base'] * weight_kg + 
               bmr_const['weight'] * height_cm - 
               bmr_const['height'] * age + 
               bmr_const['age'])
        
        # Calculate TDEE
        activity_mult = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
        tdee = bmr * activity_mult
        
        # Adjust for weight goal
        goal_adjustment = WEIGHT_GOAL_ADJUSTMENTS.get(weight_goal, 0)
        target_calories = tdee + goal_adjustment
        
        # Calculate macronutrient targets
        protein_g = weight_kg * MACRO_RATIOS['protein_per_kg']
        fat_g = target_calories * MACRO_RATIOS['fat_percentage'] / MACRO_RATIOS['fat_calories_per_g']
        remaining_calories = target_calories - (protein_g * MACRO_RATIOS['protein_calories_per_g']) - (fat_g * MACRO_RATIOS['fat_calories_per_g'])
        carbs_g = remaining_calories / MACRO_RATIOS['carb_calories_per_g']
        
        # Calculate BMI
        bmi_info = calculate_bmi(weight_kg, height_cm)
        
        # Calculate water target
        water_ml = weight_kg * WATER_ML_PER_KG
        
        targets = {
            "target_calories": round(target_calories),
            "target_protein_g": round(protein_g),
            "target_carbs_g": round(carbs_g),
            "target_fat_g": round(fat_g),
            "target_water_ml": round(water_ml),
            "bmr": round(bmr),
            "tdee": round(tdee),
            "bmi": bmi_info["bmi"],
            "bmi_classification": bmi_info["classification"],
            "healthy_weight_range": bmi_info["healthy_weight_range"]
        }
        
        # Validate targets
        validation = validate_nutrition_targets(targets)
        
        return {
            "targets": validation["targets"],
            "warnings": validation["warnings"],
            "is_valid": validation["is_valid"],
            "calculation_details": {
                "bmr": round(bmr),
                "tdee": round(tdee),
                "activity_multiplier": activity_mult,
                "goal_adjustment": goal_adjustment
            }
        }
        
    except Exception as e:
        return {"error": f"Failed to calculate targets: {str(e)}"}

@tool
def generate_smart_meal_suggestions(
    target_calories: int,
    dietary_restrictions: List[str],
    preferred_cuisines: List[str],
    cooking_time_max: int,
    meal_type: str,
    user_profile: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate intelligent meal suggestions based on targets and preferences.
    
    Args:
        target_calories: Target calories for the meal
        dietary_restrictions: List of dietary restrictions
        preferred_cuisines: List of preferred cuisines
        cooking_time_max: Maximum cooking time in minutes
        meal_type: Type of meal (breakfast, lunch, dinner, snack)
        user_profile: Optional user profile for personalization
        
    Returns:
        Dictionary with meal suggestions and analysis
    """
    global llm
    
    if llm is None:
        return {"error": "LLM not initialized"}
    
    try:
        # Create personalized prompt
        prompt = f"""
        Generate 3 meal suggestions for {meal_type} with the following criteria:
        - Target calories: approximately {target_calories}
        - Dietary restrictions: {', '.join(dietary_restrictions) if dietary_restrictions else 'None'}
        - Preferred cuisines: {', '.join(preferred_cuisines) if preferred_cuisines else 'Any'}
        - Maximum cooking time: {cooking_time_max} minutes
        
        For each suggestion, provide:
        1. Recipe name
        2. Brief description (2-3 sentences)
        3. Estimated prep time and cook time
        4. Main ingredients (5-7 items)
        5. Approximate nutritional info (calories, protein, carbs, fat)
        6. Difficulty level (Easy, Moderate, or Hard)
        7. One cooking tip
        
        Format each suggestion clearly with headers.
        Focus on balanced nutrition and realistic preparation times.
        """
        
        if user_profile:
            prompt += f"\n\nUser context: {user_profile}"
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "meal_suggestions": response.content,
            "meal_type": meal_type,
            "target_calories": target_calories,
            "criteria": {
                "dietary_restrictions": dietary_restrictions,
                "preferred_cuisines": preferred_cuisines,
                "max_cooking_time": cooking_time_max
            },
            "generation_successful": True
        }
        
    except Exception as e:
        return {"error": f"Failed to generate meal suggestions: {str(e)}"}

def initial_setup_node(state: NutritionistState) -> NutritionistState:
    """Node for initial user profile setup - let LLM handle everything naturally."""
    global llm
    
    # Only process if profile is not complete
    if state.get("profile_setup_complete", False):
        return state
    
    messages = state.get("messages", [])
    if not messages:
        return state
    
    latest_message = messages[-1]
    
    if isinstance(latest_message, HumanMessage):
        user_text = latest_message.content
        
        # Let the LLM analyze and decide what to do naturally
        analysis_prompt = f"""
        User said: "{user_text}"
        
        I'm a nutritionist and need to collect some basic information to help create meal plans.
        I need: age, weight, height, activity level, and any dietary goals/restrictions.
        
        Please respond naturally to the user. If they've provided information, acknowledge what they gave me and ask for anything missing. If they haven't provided profile info yet, ask for it in a friendly way.
        
        Be conversational and helpful.
        """
        
        try:
            response = llm.invoke([HumanMessage(content=analysis_prompt)])
            
            # Simple check: if they provided basic stats, mark profile as complete
            # Let the LLM decide this naturally too
            completion_check = f"""
            Based on this conversation:
            User: "{user_text}"
            
            Do I have enough basic information (age, weight, height) to calculate nutrition targets? 
            Answer only "YES" or "NO".
            """
            
            completion_response = llm.invoke([HumanMessage(content=completion_check)])
            profile_complete = "YES" in completion_response.content.upper()
            
            # Store the conversation context instead of extracted data
            profile = {
                "conversation_context": user_text,
                "llm_understanding": response.content
            }
            
            return {
                **state,
                "messages": [response],
                "user_profile": profile,
                "profile_setup_complete": profile_complete
            }
            
        except Exception as e:
            error_message = f"I had trouble processing that. Could you tell me your age, weight, height, and any fitness goals? Error: {str(e)}"
            return {
                **state,
                "messages": [AIMessage(content=error_message)]
            }
    
    return state

def daily_input_node(state: NutritionistState) -> NutritionistState:
    """Node for processing daily constraints and previous results naturally."""
    global llm
    
    if not state.get("profile_setup_complete", False):
        return state
    
    messages = state.get("messages", [])
    if not messages:
        return state
    
    latest_message = messages[-1]
    
    if isinstance(latest_message, HumanMessage):
        user_text = latest_message.content
        
        # Let LLM handle everything naturally
        extraction_prompt = f"""
        User said: "{user_text}"
        
        Look for and note any:
        - Cooking time available today
        - Meal preferences for today  
        - Special dietary requirements
        - Schedule constraints
        - Previous results or feedback about past meal plans
        
        Respond naturally with what you found, or say if nothing specific was mentioned.
        """
        
        try:
            extraction_result = llm.invoke([HumanMessage(content=extraction_prompt)])
            
            return {
                **state,
                "daily_constraints": {
                    "analysis": extraction_result.content,
                    "user_input": user_text
                },
                "previous_results": {
                    "analysis": extraction_result.content,
                    "user_input": user_text
                }
            }
            
        except Exception as e:
            return {
                **state,
                "daily_constraints": {"error": str(e), "raw_text": user_text},
                "previous_results": {"error": str(e), "raw_text": user_text}
            }
    
    return state

def meal_planning_node(state: NutritionistState) -> NutritionistState:
    """Generate a meal plan using natural conversation context."""
    global llm
    
    user_profile = state.get("user_profile", {})
    
    try:
        # Let the LLM create a meal plan based on the conversation context
        planning_prompt = f"""
        Based on our conversation, the user has provided: {user_profile.get('conversation_context', 'No profile info yet')}
        
        Please create a personalized daily meal plan for them. Include:
        
        1. **Daily nutrition targets** (estimated calories, protein, carbs, fat based on their profile)
        2. **Breakfast, lunch, dinner, and snack suggestions** 
        3. **Brief nutritional breakdown for each meal**
        4. **Any dietary considerations** they mentioned
        
        Be practical and helpful. If you need more information, ask for it.
        Format it nicely with clear sections.
        """
        
        response = llm.invoke([HumanMessage(content=planning_prompt)])
        
        return {
            **state,
            "meal_plan": {
                "plan": response.content,
                "created_from_context": user_profile.get('conversation_context', '')
            },
            "messages": [response]
        }
        
    except Exception as e:
        error_message = f"I had trouble creating your meal plan: {str(e)}. Could you remind me of your basic stats and goals?"
        return {
            **state,
            "messages": [AIMessage(content=error_message)]
        }
        
def nutritional_analysis_node(state: NutritionistState) -> NutritionistState:
    """Analyze recipes naturally with optional ML validation."""
    global food_predictor, llm
    
    messages = state.get("messages", [])
    if not messages:
        return state
    
    latest_message = messages[-1]
    
    if isinstance(latest_message, HumanMessage):
        user_text = latest_message.content
        
        # Let LLM analyze the recipe naturally
        analysis_prompt = f"""
        The user said: "{user_text}"
        
        Please analyze any recipe or food they mentioned. Provide:
        - Difficulty level and why
        - Meal type (breakfast/lunch/dinner/snack/dessert)
        - Estimated nutrition per serving
        - Cooking tips and time estimates
        - Any health considerations
        
        Be helpful and detailed.
        """
        
        try:
            response = llm.invoke([HumanMessage(content=analysis_prompt)])
            analysis_content = response.content
            
            # Add ML validation if available (optional)
            if food_predictor:
                try:
                    ml_validation = validate_recipe_with_ml_models(user_text, analysis_content)
                    if ml_validation.get("validation_successful"):
                        ml_data = ml_validation["ml_validation"]
                        ml_note = f"""
                        
                        ---
                        🤖 **ML Model Validation:**
                        - Difficulty: {ml_data['difficulty']['prediction']} ({ml_data['difficulty']['confidence']:.1%} confidence)
                        - Meal Type: {ml_data['meal_type']['prediction']} ({ml_data['meal_type']['confidence']:.1%} confidence)
                        - Nutrients: {ml_data['predicted_nutrients']}
                        """
                        analysis_content += ml_note
                except:
                    pass  # ML validation is optional
            
            return {
                **state,
                "messages": [AIMessage(content=analysis_content)]
            }
            
        except Exception as e:
            error_response = f"I had trouble analyzing that recipe: {str(e)}"
            return {
                **state,
                "messages": [AIMessage(content=error_response)]
            }
    
    return state

def chatbot_node(state: NutritionistState) -> NutritionistState:
    """Handle general chat interactions with enhanced context awareness."""
    global llm
    
    messages = state.get("messages", [])
    
    if not messages:
        welcome_message = """
        # 🥗 Welcome to Your AI Nutritionist! 
        
        I'm an advanced nutritionist powered by **machine learning models** that can:
        
        ## 🤖 **ML-Powered Features:**
        - 🎯 **Recipe Difficulty Prediction** - Classify recipes as Easy, More effort, or A challenge
        - 🍽️ **Meal Type Classification** - Categorize recipes (breakfast, lunch, dinner, snack, dessert)  
        - 📊 **Nutrient Prediction** - Estimate calories, protein, carbs, fat, and sodium
        - 🧮 **Smart Nutrition Calculations** - Personalized BMR, TDEE, and macro targets
        
        ## 🚀 **To Get Started:**
        Please tell me your:
        - **Age, weight, and height**
        - **Activity level** (sedentary, light, moderate, active, very active)
        - **Goal** (lose, maintain, or gain weight)
        - **Any dietary restrictions or preferences**
        
        I'll create personalized meal plans and analyze recipes using trained ML models from your food database!
        """
        return {**state, "messages": [AIMessage(content=welcome_message)]}
    
    latest_message = messages[-1]
    
    if isinstance(latest_message, HumanMessage):
        if not state.get("profile_setup_complete", False):
            # Let initial_setup_node handle this
            return state
        
        # Handle general conversation with context
        user_profile = state.get("user_profile", {})
        meal_plan = state.get("meal_plan", {})
        
        system_context = f"""
        You are a professional AI nutritionist with access to ML models for recipe analysis.
        
        User Profile: {user_profile}
        Current Meal Plan Status: {'Available' if meal_plan else 'Not created yet'}
        
        Respond helpfully to the user's message. If they ask about:
        - Recipe analysis: suggest using the ML models
        - Meal planning: offer to create a personalized plan
        - Nutrition: provide evidence-based advice
        - Their progress: ask about their experience and adjust recommendations
        
        Be encouraging, professional, and data-driven in your responses.
        """
        
        try:
            response = llm.invoke([
                HumanMessage(content=system_context),
                latest_message
            ])
            
            return {**state, "messages": [response]}
            
        except Exception as e:
            error_response = f"I'm having trouble processing your message right now: {str(e)}. Could you please try again?"
            return {**state, "messages": [AIMessage(content=error_response)]}
    
    return state

def route_based_on_state(state: NutritionistState):
    """Enhanced routing function with better decision logic."""
    messages = state.get("messages", [])
    
    # If no messages, start with chatbot (welcome)
    if not messages:
        return "chatbot"
    
    latest_message = messages[-1]
    
    # Only route on human messages
    if not isinstance(latest_message, HumanMessage):
        return "chatbot"
    
    user_text = latest_message.content.lower()
    
    # Check for profile setup
    if not state.get("profile_setup_complete", False):
        return "initial_setup"
    
    # Check for recipe analysis requests
    if any(keyword in user_text for keyword in ['analyze', 'recipe', 'difficulty', 'how hard', 'nutrients in']):
        return "nutritional_analysis"
    
    # Check for meal planning requests
    if any(keyword in user_text for keyword in ['meal plan', 'plan meals', 'what should i eat', 'create plan']):
        return "meal_planning"
    
    # Check for daily input (constraints, previous results)
    if any(keyword in user_text for keyword in ['today', 'constraint', 'time available', 'yesterday', 'result', 'felt']):
        return "daily_input"
    
    # Default to general chatbot
    return "chatbot"

def route_based_on_state_streamlit(state: NutritionistState):
    """Streamlit-specific routing function that routes to END instead of human node."""
    messages = state.get("messages", [])
    
    # If no messages, we're done (chatbot already responded)
    if not messages:
        return END
    
    latest_message = messages[-1]
    
    # Only route on human messages, otherwise end
    if not isinstance(latest_message, HumanMessage):
        return END
    
    user_text = latest_message.content.lower()
    
    # Check for profile setup
    if not state.get("profile_setup_complete", False):
        return "initial_setup"
    
    # Check for recipe analysis requests
    if any(keyword in user_text for keyword in ['analyze', 'recipe', 'difficulty', 'how hard', 'nutrients in']):
        return "nutritional_analysis"
    
    # Check for meal planning requests
    if any(keyword in user_text for keyword in ['meal plan', 'plan meals', 'what should i eat', 'create plan']):
        return "meal_planning"
    
    # Check for daily input (constraints, previous results)
    if any(keyword in user_text for keyword in ['today', 'constraint', 'time available', 'yesterday', 'result', 'felt']):
        return "daily_input"
    
    # Default to END (chatbot already responded)
    return END

def human_node(state: NutritionistState) -> NutritionistState:
    """Node for handling user input in interactive mode."""
    last_msg = state.get("messages", [])[-1] if state.get("messages") else None
    
    if last_msg and isinstance(last_msg, AIMessage):
        print("🤖 AI Nutritionist:", last_msg.content)
    
    user_input = input("\n👤 You: ")
    
    return {**state, "messages": [HumanMessage(content=user_input)]}

def build_nutritionist_graph(google_api_key: str, streamlit_mode: bool = False):
    """Build and return the nutritionist graph."""
    
    # Initialize components
    initialize_components(google_api_key)
    
    # Define the tools (ML validation is optional and called when needed)
    tools = [validate_recipe_with_ml_models, calculate_personalized_nutrition_targets, generate_smart_meal_suggestions]
    tool_node = ToolNode(tools)
    
    # Build the graph
    builder = StateGraph(NutritionistState)
    
    # Add nodes
    builder.add_node("initial_setup", initial_setup_node)
    builder.add_node("daily_input", daily_input_node)
    builder.add_node("meal_planning", meal_planning_node)
    builder.add_node("nutritional_analysis", nutritional_analysis_node)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", tool_node)
    
    if streamlit_mode:
        # Streamlit mode: Direct routing without human input node
        builder.add_edge(START, "chatbot")
        builder.add_conditional_edges("chatbot", route_based_on_state_streamlit)
        builder.add_edge("initial_setup", END)
        builder.add_edge("daily_input", "meal_planning")
        builder.add_edge("meal_planning", END)
        builder.add_edge("nutritional_analysis", END)
        builder.add_edge("tools", "chatbot")
    else:
        # Terminal mode: Include human input node
        builder.add_node("human", human_node)
        builder.add_edge(START, "chatbot")
        builder.add_edge("chatbot", "human")
        builder.add_conditional_edges("human", route_based_on_state)
        builder.add_edge("initial_setup", "human")
        builder.add_edge("daily_input", "meal_planning")
        builder.add_edge("meal_planning", "human")
        builder.add_edge("nutritional_analysis", "human")
        builder.add_edge("tools", "chatbot")
    
    return builder.compile()

def run_interactive_nutritionist(google_api_key: str):
    """Run the interactive nutritionist agent."""
    
    # Build the graph
    nutritionist_graph = build_nutritionist_graph(google_api_key)
    
    # Set initial state
    initial_state = {
        "messages": [],
        "user_profile": {},
        "weight_goal": {},
        "daily_constraints": {},
        "previous_results": {},
        "meal_plan": {},
        "nutritional_values": {},
        "profile_setup_complete": False
    }
    
    # Configuration
    config = {"recursion_limit": MAX_RECURSION_LIMIT}
    
    print("🥗 Starting AI Nutritionist Agent...")
    print("Type 'quit' to exit at any time.\n")
    
    try:
        # Run the agent
        result = nutritionist_graph.invoke(initial_state, config)
        print("\n✅ Session completed successfully!")
        return result
        
    except KeyboardInterrupt:
        print("\n👋 Session interrupted by user. Goodbye!")
        return None
    except Exception as e:
        print(f"\n❌ Session ended with error: {e}")
        return None

if __name__ == "__main__":
    # Get API key from user
    api_key = input("Please enter your Google API Key: ")
    
    if api_key:
        run_interactive_nutritionist(api_key)
    else:
        print("API key is required to run the agent.")
