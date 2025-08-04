# AI Nutritionist Agent

A sophisticated nutritionist agent built with LangGraph and Streamlit that leverages Google Gemini LLM for primary analysis and optionally uses trained machine learning models for validation and double-checking.

## Features

### 🤖 LangGraph Agent
- **State Management**: Maintains conversation context and user profile
- **Intelligent Routing**: Adaptive conversation flow based on user state and intent
- **Google Gemini Integration**: Primary analysis powered by advanced LLM
- **Optional ML Validation**: Uses trained models to double-check and validate LLM outputs

### 🍽️ Optional ML-Powered Validation
- **Recipe Difficulty Prediction**: Validates difficulty classification (Easy, More effort, A challenge)
- **Meal Type Classification**: Validates meal categorization (breakfast, lunch, dinner, snack, dessert)

### 🎯 Personalized Recommendations
- **BMR/TDEE Calculations**: Science-based calorie target calculations using Mifflin-St Jeor equation
- **Goal-Oriented Planning**: Supports weight loss, maintenance, and gain goals
- **Activity Level Adjustment**: Accounts for sedentary to very active lifestyles
- **Dietary Restrictions**: Handles various dietary preferences and restrictions

### 🖥️ Streamlit Interface
- **Interactive Chat**: Natural conversation with the AI nutritionist
- **Visual Dashboard**: Charts and metrics for nutrition tracking
- **Meal Planning**: Generate and modify daily meal plans
- **Recipe Analyzer**: Analyze individual recipes with ML validation (difficulty & meal type only)
- **ML Toggle**: Enable/disable ML validation as needed

## Architecture Philosophy

### Primary LLM Analysis + Optional ML Validation

The agent uses a **two-layer approach**:

1. **Primary Layer (Google Gemini)**: 
   - Handles all main conversation and analysis
   - Generates meal plans, provides nutrition advice
   - Analyzes recipes with contextual understanding

2. **Validation Layer (ML Models - Optional)**:
   - Provides additional validation for recipe analysis
   - Double-checks difficulty and meal type predictions
   - Can be enabled/disabled based on preference

This approach ensures the agent works smoothly even without ML models while providing enhanced validation for difficulty and meal type when available.

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

The app includes:
- 🤖 **ML Validation Toggle**: Enable/disable ML model validation
- 💬 **Chat Interface**: Natural conversation with the AI nutritionist
- 🍽️ **Meal Planning**: Complete daily meal plan generation
- 📊 **Dashboard**: Nutrition tracking and visualization

### Running the Demo

```bash
python demo.py
```

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

### Optional ML Integration
The agent can optionally use your existing ML models for validation:
- **Difficulty Model**: Validates cooking difficulty predictions
- **Meal Type Model**: Validates meal category classifications

### Tool Functions
- `validate_recipe_with_ml_models()`: ML validation of recipes (difficulty & meal type only)
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
```

### Supported Model Types
- XGBoost (`.pt` files)
- LightGBM (`.pt` files)
- PyTorch MLP (`.pt` files)

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
Agent: LLM analyzes recipe → ML validation (difficulty & meal type) → Combined response
```

### 4. Meal Plan Generation
```
User: "Create a meal plan for today"
Agent: Calculates targets → Generates breakfast/lunch/dinner/snacks → Provides complete plan
```

## ML Validation Features

When enabled, ML validation provides:

- **Confidence Scores**: ML model confidence levels for predictions
- **Comparison**: Side-by-side comparison with LLM analysis
- **Consistency Check**: Verification of difficulty and meal type classifications

Example ML validation output:
```
🤖 ML Validation Results (for double-checking):
- Difficulty: Easy (confidence: 85%)
- Meal Type: dinner (confidence: 92%)
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

### Disabling ML Validation
In Streamlit: Uncheck the "🤖 Enable ML Validation" toggle
In code: Set `ml_validation_enabled = False`

### Customizing Nutrition Calculations
Modify constants in `config.py`:
- `BMR_CONSTANTS`: Basal metabolic rate calculation parameters
- `ACTIVITY_MULTIPLIERS`: Activity level multipliers
- `MACRO_RATIOS`: Macronutrient distribution ratios

## Troubleshooting

### Without ML Models
The agent works perfectly without ML models - it will simply skip validation and rely entirely on the powerful Gemini LLM for analysis (no difficulty/meal type double-check).

### API Issues
- Verify your Google API key is valid and has Gemini access
- Check API quota and usage limits
- Ensure stable internet connectivity

### Performance
- Enable ML validation only when needed for better performance
- ML models require moderate RAM (500MB-2GB depending on model size)
- Consider model quantization for production deployment

## Future Enhancements

### Planned Features
- **Multi-day Planning**: Plan entire weeks with ML optimization
- **Shopping List Generation**: Automatic grocery lists from meal plans
- **Progress Tracking**: Integration with fitness apps and wearables
- **Social Features**: Share meal plans and get community feedback
- **Advanced Analytics**: Detailed nutrition insights with ML trends

### ML Improvements
- **Ensemble Validation**: Combine multiple model predictions
- **Ingredient-Level Analysis**: More granular recipe understanding
- **Cuisine-Specific Models**: Better cultural food understanding
- **Personalized Models**: Adapt to individual user preferences over time

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is part of the larger food prediction system. Please refer to the main project license.
