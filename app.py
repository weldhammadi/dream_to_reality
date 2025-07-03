# app.py

import streamlit as st
from backend import (
    speach_to_text, 
    generate_image_prompt, 
    generate_image_with_clipdrop,
    analyze_content_emotions,
    analyze_content_themes,
    init_database,
    save_to_history,
    get_history,
    delete_from_history
)
import tempfile
import os
from datetime import datetime
import json

# Initialize database
init_database()

st.set_page_config(page_title="Audio to Image Generator", page_icon="🎨", layout="wide")

st.title("🎨 Audio to Image Generator")
st.markdown("Upload an audio file or record audio to generate an image based on your description!")

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["🎨 Generate", "📊 Content Analysis", "📜 History"])

with tab1:
    st.header("🎨 Generate Image from Audio")
    
    # Audio file uploader
    uploaded_audio = st.file_uploader("🎵 Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])

    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/wav")
        
        if st.button("🎯 Generate Image from Audio", key="generate_image"):
            with st.spinner("Processing audio and generating image..."):
                try:
                    # Step 1: Transcribe audio
                    st.info("Step 1: Transcribing audio...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_file:
                        temp_file.write(uploaded_audio.read())
                        temp_file_path = temp_file.name
                    
                    transcribed_text = speach_to_text(temp_file_path, language="fr")
                    os.unlink(temp_file_path)
                    
                    if not transcribed_text:
                        st.error("Failed to transcribe audio. Please ensure the file contains clear speech.")
                        st.stop()
                    
                    st.success("✅ Audio transcribed successfully!")
                    st.markdown("### 📝 Transcribed text:")
                    st.write(transcribed_text)
                    
                    # Step 2: Analyze content
                    st.info("Step 2: Analyzing content...")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 😊 Emotional Analysis")
                        emotion_analysis = analyze_content_emotions(transcribed_text)
                        
                        # Display emotion analysis as progress bars
                        for emotion, score in emotion_analysis.items():
                            st.progress(score, text=f"{emotion}: {score:.2f}")
                    
                    with col2:
                        st.markdown("#### 🎯 Theme Analysis")
                        theme_analysis = analyze_content_themes(transcribed_text)
                        
                        # Display theme analysis as progress bars
                        for theme, score in theme_analysis.items():
                            st.progress(score, text=f"{theme}: {score:.2f}")
                    
                    # Step 3: Generate image prompt
                    st.info("Step 3: Generating image prompt...")
                    image_prompt = generate_image_prompt(transcribed_text)
                    
                    st.success("✅ Image prompt generated!")
                    st.markdown("### 🖼️ Generated prompt:")
                    st.write(image_prompt)
                    
                    # Step 4: Generate image with ClipDrop
                    st.info("Step 4: Generating image with ClipDrop...")
                    image_data = generate_image_with_clipdrop(image_prompt)
                    
                    if image_data:
                        st.success("✅ Image generated successfully!")
                        
                        # Display the generated image
                        st.markdown("### 🎨 Generated Image:")
                        st.image(image_data, caption="Generated Image", use_container_width=True)
                        
                        # Save to history
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"generated_image_{timestamp}.png"
                        image_path = os.path.join("generated_images", filename)
                        
                        # Create directory if it doesn't exist
                        os.makedirs("generated_images", exist_ok=True)
                        
                        # Save image to file
                        with open(image_path, "wb") as f:
                            f.write(image_data)
                        
                        # Save to history database
                        content_analysis = {"emotions": emotion_analysis, "themes": theme_analysis}
                        generation_id = save_to_history(
                            transcribed_text, 
                            emotion_analysis, 
                            image_prompt, 
                            image_path, 
                            content_analysis
                        )
                        
                        st.success(f"✅ Saved to history with ID: {generation_id}")
                        
                        # Provide download button
                        st.download_button(
                            label="💾 Download Image",
                            data=image_data,
                            file_name=filename,
                            mime="image/png"
                        )
                    else:
                        st.error("Failed to generate image.")
                        
                except Exception as e:
                    st.error(f"Error during processing: {e}")
    else:
        st.info("Please upload an audio file to begin!")

with tab2:
    st.header("📊 Content Analysis Only")
    st.markdown("Analyze the content and emotions of your audio without generating images.")
    
    # Audio file uploader for analysis only
    uploaded_audio_analysis = st.file_uploader("🎵 Upload an audio file for analysis", type=["wav", "mp3", "m4a", "ogg"], key="analysis_audio")

    if uploaded_audio_analysis is not None:
        st.audio(uploaded_audio_analysis, format="audio/wav")
        
        if st.button("📊 Analyze Content", key="analyze_content"):
            with st.spinner("Analyzing content..."):
                try:
                    # Transcribe audio
                    st.info("Transcribing audio...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_file:
                        temp_file.write(uploaded_audio_analysis.read())
                        temp_file_path = temp_file.name
                    
                    transcribed_text = speach_to_text(temp_file_path, language="fr")
                    os.unlink(temp_file_path)
                    
                    if not transcribed_text:
                        st.error("Failed to transcribe audio.")
                        st.stop()
                    
                    st.success("✅ Audio transcribed successfully!")
                    st.markdown("### 📝 Transcribed text:")
                    st.write(transcribed_text)
                    
                    # Analyze content
                    st.info("Analyzing content...")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 😊 Emotional Analysis")
                        emotion_analysis = analyze_content_emotions(transcribed_text)
                        
                        # Display as chart
                        st.bar_chart(emotion_analysis)
                        
                        # Display detailed scores
                        for emotion, score in emotion_analysis.items():
                            st.metric(label=emotion.replace("_", " ").title(), value=f"{score:.3f}")
                    
                    with col2:
                        st.markdown("#### 🎯 Theme Analysis")
                        theme_analysis = analyze_content_themes(transcribed_text)
                        
                        # Display as chart
                        st.bar_chart(theme_analysis)
                        
                        # Display detailed scores
                        for theme, score in theme_analysis.items():
                            st.metric(label=theme.replace("_", " ").title(), value=f"{score:.3f}")
                    
                    # Display combined analysis
                    st.markdown("### 📋 Complete Analysis")
                    analysis_data = {
                        "transcribed_text": transcribed_text,
                        "emotions": emotion_analysis,
                        "themes": theme_analysis
                    }
                    
                    st.json(analysis_data)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")

with tab3:
    st.header("📜 Generation History")
    st.markdown("View and manage your previous generations.")
    
    # Controls
    col1, col2 = st.columns([3, 1])
    with col1:
        limit = st.slider("Number of items to display", 1, 50, 10)
    with col2:
        if st.button("🔄 Refresh History"):
            st.rerun()
    
    # Get and display history
    history = get_history(limit)
    
    if history:
        for i, item in enumerate(history):
            with st.expander(f"🎨 Generation {i+1} - {item['timestamp'][:19]}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Transcribed Text:**")
                    st.write(item['transcribed_text'])
                    
                    st.markdown("**Generated Prompt:**")
                    st.write(item['generated_prompt'])
                    
                    if item['emotion_analysis']:
                        st.markdown("**Emotion Analysis:**")
                        emotion_data = {k: v for k, v in item['emotion_analysis'].items()}
                        st.bar_chart(emotion_data)
                    
                    if item['content_analysis'] and 'themes' in item['content_analysis']:
                        st.markdown("**Theme Analysis:**")
                        theme_data = {k: v for k, v in item['content_analysis']['themes'].items()}
                        st.bar_chart(theme_data)
                
                with col2:
                    # Display image if it exists
                    if item['image_path'] and os.path.exists(item['image_path']):
                        st.image(item['image_path'], caption="Generated Image", use_container_width=True)
                        
                        # Download button
                        with open(item['image_path'], "rb") as f:
                            st.download_button(
                                label="💾 Download",
                                data=f.read(),
                                file_name=os.path.basename(item['image_path']),
                                mime="image/png",
                                key=f"download_{item['id']}"
                            )
                    else:
                        st.info("Image not available")
                    
                    # Delete button
                    if st.button("🗑️ Delete", key=f"delete_{item['id']}"):
                        delete_from_history(item['id'])
                        if item['image_path'] and os.path.exists(item['image_path']):
                            os.remove(item['image_path'])
                        st.success("Item deleted!")
                        st.rerun()
    else:
        st.info("No generation history found. Create your first generation in the Generate tab!")

# Sidebar with information
st.sidebar.markdown("## 🔧 How it works")
st.sidebar.markdown("""
1. **Upload Audio**: Upload an audio file describing what you want to create
2. **Transcription**: Your audio is transcribed using Whisper AI
3. **Content Analysis**: AI analyzes emotions and themes in your description
4. **Prompt Generation**: Mistral AI creates a detailed image prompt
5. **Image Generation**: ClipDrop generates the final image
6. **History**: All generations are saved for future reference
""")

st.sidebar.markdown("## 💡 Tips")
st.sidebar.markdown("""
- Speak clearly and describe your vision in detail
- Use French or English for best results
- Be specific about colors, styles, and composition
- Check the Content Analysis tab to understand how AI interprets your words
- Use History to revisit and download previous generations
""")

st.sidebar.markdown("## 📊 Features")
st.sidebar.markdown("""
- **Emotion Analysis**: Detects happiness, anxiety, sadness, anger, fatigue, fear
- **Theme Analysis**: Identifies nature, urban, people, objects, abstract, action, calm themes
- **Complete History**: All generations saved with analysis data
- **Download & Delete**: Manage your generated images easily
""")
