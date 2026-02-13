import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go

# Настройка страницы
st.set_page_config(
    page_title="News Classifier",
    page_icon="📰",
    layout="wide"
)

# Заголовок
st.title("📰 News Classification Demo")
st.markdown("*Automatic categorization of news articles using BERT*")
st.markdown("---")

# Загрузка модели с кэшированием
@st.cache_resource
def load_model():
    """Загружает модель один раз и кэширует"""
    try:
        # Попробовать загрузить локальную модель
        model = AutoModelForSequenceClassification.from_pretrained('final_model_bert')
        tokenizer = AutoTokenizer.from_pretrained('final_model_bert')
        st.success("✅ Local model loaded successfully!")
    except:
        # Если локальной нет, использовать предобученную из HuggingFace
        st.warning("⚠️ Local model not found. Loading pre-trained model from HuggingFace...")
        model = AutoModelForSequenceClassification.from_pretrained('fabriceyhc/bert-base-uncased-ag_news')
        tokenizer = AutoTokenizer.from_pretrained('fabriceyhc/bert-base-uncased-ag_news')
        st.success("✅ Pre-trained model loaded!")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

# Загрузка модели
with st.spinner('Loading model...'):
    model, tokenizer, device = load_model()

label_names = ['World', 'Sports', 'Business', 'Sci/Tech']
label_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
label_emojis = ['🌍', '⚽', '💼', '🔬']

def classify_news(text):
    """Классификация текста"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
    
    pred_label = label_names[pred_idx]
    confidence = probs[0][pred_idx].item()
    all_probs = {label_names[i]: probs[0][i].item() for i in range(4)}
    
    return pred_label, confidence, all_probs

# Sidebar с примерами
st.sidebar.header("📝 Example Articles")
st.sidebar.markdown("Click to try:")

examples = {
    "🔬 Tech": "Apple announces new iPhone with advanced AI capabilities and improved camera system.",
    "⚽ Sports": "LeBron James scores 40 points as Lakers beat Celtics in NBA finals game.",
    "💼 Business": "Stock markets rise as Federal Reserve signals potential interest rate cuts.",
    "🌍 World": "President Biden meets European leaders to discuss Ukraine crisis in Brussels.",
    "🔬 Science": "NASA discovers water ice on Mars surface raising possibilities for colonization.",
}

selected_example = st.sidebar.radio("Select example:", list(examples.keys()))

if st.sidebar.button("📋 Use This Example"):
    st.session_state.input_text = examples[selected_example]

# Основной интерфейс
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter News Article")
    
    # Текстовое поле
    default_text = st.session_state.get('input_text', '')
    user_input = st.text_area(
        "Paste or type your news article here:",
        value=default_text,
        height=150,
        placeholder="Enter news text here..."
    )
    
    classify_button = st.button("🚀 Classify", type="primary", use_container_width=True)

with col2:
    st.subheader("Categories")
    for emoji, label in zip(label_emojis, label_names):
        st.markdown(f"{emoji} **{label}**")

# Классификация
if classify_button and user_input.strip():
    if len(user_input.strip()) < 10:
        st.error("⚠️ Please enter at least 10 characters")
    else:
        with st.spinner('Classifying...'):
            pred_label, confidence, all_probs = classify_news(user_input)
        
        # Результат
        st.markdown("---")
        st.subheader("📊 Classification Result")
        
        # Основной результат
        pred_emoji = label_emojis[label_names.index(pred_label)]
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.metric("Category", f"{pred_emoji} {pred_label}")
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        with col3:
            confidence_color = "🟢" if confidence > 0.9 else "🟡" if confidence > 0.7 else "🔴"
            st.metric("Status", confidence_color)
        
        # График вероятностей
        st.markdown("### 📈 All Probabilities")
        
        # Сортируем по вероятности
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Создаем горизонтальный bar chart с Plotly
        fig = go.Figure()
        
        for label, prob in sorted_probs:
            idx = label_names.index(label)
            emoji = label_emojis[idx]
            color = label_colors[idx]
            
            fig.add_trace(go.Bar(
                y=[f"{emoji} {label}"],
                x=[prob * 100],
                orientation='h',
                marker=dict(color=color),
                text=[f"{prob:.1%}"],
                textposition='auto',
                hovertemplate=f"<b>{label}</b><br>Probability: {prob:.2%}<extra></extra>"
            ))
        
        fig.update_layout(
            showlegend=False,
            xaxis_title="Probability (%)",
            yaxis_title="",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Детальная таблица
        with st.expander("📋 Detailed Probabilities"):
            for label, prob in sorted_probs:
                idx = label_names.index(label)
                emoji = label_emojis[idx]
                col1, col2, col3 = st.columns([2, 2, 3])
                with col1:
                    st.write(f"{emoji} **{label}**")
                with col2:
                    st.write(f"{prob:.4f}")
                with col3:
                    st.progress(prob)

elif classify_button:
    st.warning("⚠️ Please enter some text first")

# Информация внизу
st.markdown("---")
st.markdown("""
### ℹ️ About This Demo

This application uses a **BERT-based model** fine-tuned on the AG News dataset to classify news articles into 4 categories:
- 🌍 **World**: International events, politics, conflicts
- ⚽ **Sports**: Athletic competitions, teams, players
- 💼 **Business**: Companies, markets, economy, finance
- 🔬 **Sci/Tech**: Technology, science, innovations

**Model Performance**: ~94% accuracy on test set

**Note**: Model may make errors on ambiguous articles, especially between Business and Sci/Tech categories.
""")

# Статистика в sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info(f"""
**Architecture**: BERT-base  
**Parameters**: 110M  
**Training**: AG News dataset  
**Accuracy**: 94.0%  
**Device**: {device}
""")
