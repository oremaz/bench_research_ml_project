# AI Nutritionist Agent

A sophisticated nutritionist agent built with LangGraph and Streamlit that leverages Google Gemini LLM for primary analysis and, if available, uses trained machine learning models for validation and double-checking. Includes secure user registration, session management, and advanced personalization features.

## Features

### 🤖 LangGraph Agent
- **State Management**: Maintains conversation context and user profile
- **Intelligent Routing**: Adaptive conversation flow based on user state and intent
- **Google Gemini Integration**: Primary analysis powered by advanced LLM
- **Automatic ML Validation**: Uses trained models to double-check and validate LLM outputs if models are present (no toggle)

### 🍽️ Optional ML-Powered Validation
- **Recipe Difficulty Prediction**: Validates difficulty classification (Easy, More effort, A challenge)
- **Meal Type Classification**: Validates meal categorization (breakfast, lunch, dinner, snack, dessert)
- **Total Time Classification**: Validates total cooking time class (e.g., under 30 min, 30-60 min, over 60 min)

### 🎯 Personalized Recommendations
- **BMR/TDEE Calculations**: Science-based calorie target calculations using Mifflin-St Jeor equation
- **Goal-Oriented Planning**: Supports weight loss, maintenance, and gain goals
- **Activity Level Adjustment**: Accounts for sedentary to very active lifestyles
- **Dietary Restrictions**: Handles various dietary preferences and restrictions

### 🖥️ Streamlit Interface
- **Interactive Chat**: Natural conversation with the AI nutritionist
- **Visual Dashboard**: Charts and metrics for nutrition tracking
- **Meal Planning**: Generate and modify daily meal plans
- **Recipe Analyzer**: Analyze individual recipes with ML validation (difficulty, meal type & total time class)
- **Registration & Login**: Secure onboarding, persistent user profiles, and chat history
- **Session Management**: Multiple chat sessions per user, with memory/context selection for continuity
- **Standalone Recipe Analyzer**: Analyze any recipe with ML/LLM in a dedicated tool
- **Daily Results Tracker**: Log and analyze daily progress and meal compliance
- **Memory Selection**: Use previous chat sessions as context for continuity and personalization

## Architecture Philosophy

### Primary LLM Analysis + Automatic ML Validation

The agent uses a **two-layer approach**:

1. **Primary Layer (Google Gemini)**: 
   - Handles all main conversation and analysis
   - Generates meal plans, provides nutrition advice
   - Analyzes recipes with contextual understanding

2. **Validation Layer (ML Models - Automatic)**:
   - Provides additional validation for recipe analysis
   - Double-checks difficulty, meal type, and total time class predictions
   - ML validation is always performed if models are present; if not, the agent relies on LLM analysis only.

There is no toggle to enable/disable ML validation—if models are available, validation is automatic. The agent works smoothly even without ML models.

## Installation

1. **Clone and navigate to the project**:
   ```bash
   cd /path/to/your/food/project/nut_agent
   ```

2. **Run the setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Set up environment**:
   - Add your Google API key to the `.env` file
   - Ensure your trained ML models are available in `../food_preds/results/` (optional)

## Usage

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```


#### Streamlit App Features
- 💬 **Chat Interface**: Natural conversation with the AI nutritionist
- 🍽️ **Meal Planning**: Complete daily meal plan generation
- 📊 **Dashboard**: Nutrition tracking and visualization
- 📝 **Registration & Login**: Secure onboarding, persistent user profiles, and chat history
- 🧠 **Session Memory**: Select previous chat sessions as context for continuity
- 🍳 **Standalone Recipe Analyzer**: Analyze any recipe with ML/LLM in a dedicated tool
- 📈 **Daily Results Tracker**: Log and analyze daily progress and meal compliance


### Using the Agent Programmatically

```python
from improved_agent import build_nutritionist_graph, initialize_components
from langchain_core.messages import HumanMessage, AIMessage

# Initialize the agent
initialize_components("your_google_api_key")
agent_graph = build_nutritionist_graph("your_google_api_key")

# Start a conversation
state = {
    "messages": [HumanMessage(content="I'm 25, 70kg, 175cm, male, moderately active. I want to lose weight.")],
    "user_profile": {},
    "profile_setup_complete": False,
    # ... other state fields
}

# Process the conversation
result = agent_graph.invoke(state, {"recursion_limit": 50})

# Get the response
for message in reversed(result["messages"]):
    if isinstance(message, AIMessage):
        print(message.content)
        break
```

## Architecture

### LangGraph State Management
```python
class NutritionistState(TypedDict):
    messages: Annotated[list, add_messages]
    user_profile: Dict[str, Any]
    weight_goal: Dict[str, Any]
    daily_constraints: Dict[str, Any]
    previous_results: Dict[str, Any]
    meal_plan: Dict[str, Any]
    nutritional_values: Dict[str, Any]
    profile_setup_complete: bool
```

### Conversation Flow
1. **Initial Setup**: Extract user profile (age, weight, height, goals)
2. **Daily Input**: Process constraints and previous day feedback
3. **Meal Planning**: Generate personalized daily meal plans
4. **Recipe Analysis**: Analyze recipes with optional ML validation
5. **General Chat**: Handle questions and provide nutrition advice

### ML Integration
The agent will automatically use your existing ML models for validation if they are present:
- **Difficulty Model**: Validates cooking difficulty predictions
- **Meal Type Model**: Validates meal category classifications
- **Total Time Model**: Validates total cooking time class (e.g., under 30 min, 30-60 min, over 60 min)

### Tool Functions
- `validate_recipe_with_ml_models()`: ML validation of recipes (difficulty, meal type & total time class)
- `calculate_personalized_nutrition_targets()`: Calculates BMR/TDEE and targets
- `generate_smart_meal_suggestions()`: Creates meal recommendations

## Configuration

### Model Paths (Optional)
If ML validation is enabled, the agent looks for models in:
```
../food_preds/results/
├── difficulty_train/
│   ├── xgboost_classifier.pt
│   ├── lightgbm_classifier.pt
│   └── mlp_classifier.pt
├── meal_train/
│   └── [similar structure]
├── total_time_class_train/
│   └── lightgbm_classifier.pt
```

### Supported Model Types
- XGBoost (`.pt` files)
- LightGBM (`.pt` files)
- PyTorch MLP (`.pt` files)
- (All model types above are supported for difficulty, meal type, and total time classification)

## Example Workflows

### 1. Profile Setup
```
User: "I'm 28, female, 65kg, 168cm, lightly active, want to lose 5kg"
Agent: Extracts profile → Calculates BMI and targets → Sets up personalized state
```

### 2. Daily Planning
```
User: "I have 30 minutes for cooking today, prefer Mediterranean food"
Agent: Generates meal plan → Distributes calories → Suggests specific recipes
```

### 3. Recipe Analysis with ML Validation
```
User: "How difficult is grilled salmon with quinoa?"
Agent: LLM analyzes recipe → ML validation (difficulty, meal type & total time class) → Combined response
```

### 4. Meal Plan Generation
```
User: "Create a meal plan for today"
Agent: Calculates targets → Generates breakfast/lunch/dinner/snacks → Provides complete plan
```

## ML Validation Features

When ML models are present, validation provides:

- **Confidence Scores**: ML model confidence levels for predictions
- **Comparison**: Side-by-side comparison with LLM analysis
- **Consistency Check**: Verification of difficulty, meal type, and total time classifications

Example ML validation output:
```
🤖 ML Validation Results (for double-checking):
- Difficulty: Easy (confidence: 85%)
- Meal Type: dinner (confidence: 92%)
- Total Time Class: under 30 min (confidence: 88%)
```

## Customization

### Adding New Tools
```python
@tool
def your_custom_tool(param: str) -> Dict[str, Any]:
    """Your tool description."""
    # Your tool logic
    return result

# Add to tools list in improved_agent.py
tools = [validate_recipe_with_ml_models, calculate_personalized_nutrition_targets, your_custom_tool]
```


### Customizing Nutrition Calculations and Extending the Agent
- **Nutrition Science**: Modify constants in `config.py`:
   - `BMR_CONSTANTS`: Basal metabolic rate calculation parameters
   - `ACTIVITY_MULTIPLIERS`: Activity level multipliers
   - `MACRO_RATIOS`: Macronutrient distribution ratios
- **Add New Tools**: Use the `@tool` decorator in `improved_agent.py` and add to the tool list.
- **Add/Change Models**: Update model paths and registry in `config.py` and `model_predictor.py`.
- **UI Customization**: Add new features or tabs in `streamlit_app.py`.

## Troubleshooting

### Without ML Models
The agent works perfectly without ML models - it will simply skip validation and rely entirely on the powerful Gemini LLM for analysis (no difficulty/meal type double-check).

### API Issues
- Verify your Google API key is valid and has Gemini access
- Check API quota and usage limits
- Ensure stable internet connectivity

### Performance
- ML models require moderate RAM (500MB-2GB depending on model size)
- Consider model quantization for production deployment

## User Registration, Login, and Session Management

- **Registration & Login**: Users create accounts with username and password (passwords are securely hashed).
- **Profile Data**: Registration collects age, weight, height, gender, activity level, goals, dietary preferences, cooking experience, budget, meal schedule, health conditions, and more.
- **Session Management**: Each user can have multiple chat sessions, with the ability to select previous sessions as memory/context for new chats.
- **Chat History**: All chat history and user data are stored securely in the `secrets/` directory (excluded from version control).
- **Export/Import**: Users can export their conversation history as JSON.
- **Security**: All sensitive data is stored in `secrets/`, and passwords are hashed.

## Streamlit App Advanced Features

- **Standalone Recipe Analyzer**: Analyze any recipe with ML/LLM in a dedicated tool, with detailed breakdown and confidence scores.
- **Daily Results Tracker**: Log daily weight, meal compliance, and additional notes; receive AI analysis and recommendations.
- **Memory Selection**: Select previous chat sessions to use as context for continuity and personalization.
- **Quick Actions**: One-click buttons for meal plan generation, nutrition target calculation, and more.
- **Debug & Export**: View session/debug info and export conversation data.

## Security and Data Handling

- All user data and chat logs are stored in the `secrets/` directory and are excluded from version control for privacy.
- Passwords are securely hashed.
- Users can export their conversation history as JSON.
- Registration and chat data are never shared or uploaded.

## Future Enhancements

### Planned Features
- **Multi-day Planning**: Plan entire weeks with ML optimization
- **Shopping List Generation**: Automatic grocery lists from meal plans
- **Progress Tracking**: Integration with fitness apps and wearables
- **Social Features**: Share meal plans and get community feedback
- **Advanced Analytics**: Detailed nutrition insights with ML trends

### ML Improvements (Planned)
- **Ensemble Validation**: Combine multiple model predictions (future)
- **Ingredient-Level Analysis**: More granular recipe understanding (future)
- **Cuisine-Specific Models**: Better cultural food understanding (future)
- **Personalized Models**: Adapt to individual user preferences over time (future)