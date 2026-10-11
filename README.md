# Pupil Tracking @ LARA

Research log for my pupil tracking work at the **Laboratory for AI, Robotics and Automation (LARA)**, UC Davis.

> The working code for this project lives in a private repository while the research is in progress. This repo documents the problem, approach, and system design at a public-safe level.

---

## The Problem

Eye tracking is a powerful input for robotics and human-in-the-loop control. Knowing where someone is looking tells a system what they care about, often before they act. But reliable eye tracking usually depends on specific hardware: infrared illumination, tuned cameras, and controlled lighting.

This project focuses on **robust pupil detection from grayscale camera feeds**. The goal is accurate, low-latency pupil localization despite noise, reflections, eyelid occlusion, and changing lighting, so gaze can be used as a real-time signal in a closed-loop robotic system.

## Approach

I'm surveying and benchmarking three families of methods:

| Family | Idea | Trade-off |
|---|---|---|
| **Traditional CV** | Thresholding, edge detection, and ellipse fitting to find the dark pupil region | Fast and interpretable, but brittle under reflections, blur, and occlusion |
| **Hybrid** | Classical pipelines refined or gated by lightweight learned components | Keeps classical speed while recovering from its common failure cases |
| **Deep learning** | CNN-based segmentation / landmark models trained end-to-end on eye images | Most robust to messy conditions, but heavier and harder to run in real time |

The comparison centers on what matters for live control: **accuracy, robustness, and latency**.

## System Context

Pupil tracking is one part of a larger real-time pipeline:

```
 Pupil Core eye tracker
          │  (gaze + pupil data)
          ▼
 Python middleware  ──  ZeroMQ / UDP sockets
          │  (low-latency message bridge)
          ▼
 LabVIEW control system
          │
          ▼
 Closed-loop robot control
```

I built the middleware that bridges **Pupil Core** to **LabVIEW** over ZeroMQ and UDP, so eye-tracking data can drive the control loop in real time.

## Tech

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![ZeroMQ](https://img.shields.io/badge/ZeroMQ-DF0000?style=flat-square&logo=zeromq&logoColor=white)
![LabVIEW](https://img.shields.io/badge/LabVIEW-FFDB00?style=flat-square&logoColor=black)

Pupil Core (Pupil Labs) · ZeroMQ · UDP sockets · LabVIEW

## Status

- [x] Middleware bridging Pupil Core to LabVIEW over ZeroMQ/UDP
- [x] Survey of traditional, hybrid, and deep learning pupil detection methods
- [ ] Benchmarking on grayscale camera data
- [ ] Integrating the selected detector into the closed-loop system

---

**Raquib Alam** · ML Researcher @ LARA, UC Davis · [GitHub](https://github.com/ChiefGuap) · [LinkedIn](https://www.linkedin.com/in/chiefguap)
