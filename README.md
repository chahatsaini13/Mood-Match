🎨 Emotion-Based Moodboard Generator

An AI-powered moodboard generator designed to spark inspiration for designers, students, and content creators. Users describe how they feel—and this tool curates a visual board of fonts, colors, and images tailored to their emotion.

---

🧠 How It Works

1. **User Prompt:** User inputs a sentence or phrase describing their mood or intent.
2. **Emotion Detection:** A machine learning model analyzes the prompt and predicts the underlying emotion.
3. **Style Mapping:** The detected emotion is mapped to a predefined style guide (JSON) that includes:
   - Fonts
   - Color palettes
   - Image/pattern keywords
4. **Asset Curation:** Relevant images and patterns are pulled from APIs like Unsplash, Pinterest, and Pexels.
5. **Moodboard Assembly:** The elements are placed into a consistent layout template to create a unique, emotion-aligned visual.

---

🧩 Tech Stack

- **Frontend:** HTML, CSS, Javascript
- **Backend:** Python / Node.js (for API calls and generation pipeline)
- **ML Model:** Custom-trained emotion classifier (text-to-emotion)
- **Image Sources:** Unsplash API, Pexels API, Pinterest scraping
- **Design Rendering:** Figma, custom SVG templating
- **Versioning & Monitoring:** Weights & Biases
