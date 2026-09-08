# 🤟 SignBridge ISL

> **Real-Time, Offline-Capable Indian Sign Language (ISL) Assistive Communication System.**

Over 7 million Deaf and Hard-of-Hearing individuals in India rely on **Indian Sign Language (ISL)** as their primary mode of communication. Due to a severe shortage of certified ISL interpreters, a massive everyday communication gap exists between ISL signers and the non-signing majority. 

**SignBridge ISL** bridges this gap using deep learning and modern web engineering. It provides an accessible, low-latency, and offline-capable translation pipeline that converts ISL gestures into spoken words and fluent regional text—without requiring continuous cloud connectivity.

---

## 🌟 Key Features

* **✋ Indian Sign Language (ISL) Gesture Translation:**
  * **MediaPipe Landmark Tracking:** Tracks 258 spatial body keypoints across Pose, Left Hand, and Right Hand markers in real time.
  * **3-Layer LSTM Network:** Classifies dynamic dynamic gestural sequences over rolling 30-frame time steps.
  * **Edge-Optimized Offline Pipeline:** Runs entirely on-device for core computer vision and speech rendering, making it functional in low-connectivity regions.

* **🧠 Smart Sentence Construction & Multi-Language Support:**
  * **Sign Gloss-to-Grammar Pipeline:** Converts disjointed fingerspelling and isolated ISL sign glosses into coherent, natural conversational sentences.
  * **Multilingual Translation:** Translates generated English text into Indian regional languages (e.g., Tamil, Hindi) via built-in translation modules.

* **🎤 Bidirectional Speech-to-Sign Playback:**
  * **Voice/Text Driven ISL Avatars:** Converts spoken Hindi/English input into sequenced ISL video animations for non-signers communicating back to Deaf users.
  * **Fallback Dynamic Fingerspelling:** Automatically breaks down unindexed words into individual ISL alphabetical sign sequences.

* **💻 High-Performance Engineering:**
  * **Decoupled HTML5 Canvas Feed:** Streams skeleton and live video feeds via Server-Sent Events (SSE) to eliminate stream stuttering and keep latency under 50ms.
  * **Real-time Analytics Dashboard:** Displays softMax prediction probabilities, confidence waves, global gesture cooldown bars, and skeleton overlay toggles.

---

## 🏗 System Architecture

```text
       ┌────────────────────────────────────────────────────────┐
       │             Real-Time Input Capture (Camera)           │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     MediaPipe Holistic (Extracts 258 Spatial Keypoints) │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     3-Layer Deep LSTM Sequence Model (ISL Recognition) │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │        Stability & Cooldown Buffer Engine              │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │   Smart Sentence Smoothing Engine (Gloss ➔ Grammar)    │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │    SSE Stream ──► Web UI Canvas ──► Regional TTS      │
       └────────────────└───────────────────────────────────────┘
