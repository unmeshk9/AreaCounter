# AreaCounter

A C#/.NET application for detecting and counting rectangular areas in raster images using a hybrid approach that combines traditional image processing (OpenCV) and machine learning (ML.NET).

---

## Features

- **Hybrid Rectangle Detection:** Leverages both classical image processing and ML regression for robust rectangle counting.
- **Feature Extraction:** Extracts contour statistics, edge density, and aspect ratio features from images.
- **ML Model Training:** Train a regression model using your own labeled images.
- **CLI Interface:** Easy-to-use command-line interface for both training and prediction.
- **Extensible & Testable:** Modular code with a comprehensive unit test suite (xUnit).

---

## Getting Started

### Prerequisites
- [.NET 8.0 SDK](https://dotnet.microsoft.com/download)
- Windows (OpenCvSharp4.runtime.win is used)

### Installation
1. **Clone the repository:**
   ```
   git clone <your-repo-url>
   cd AreaCounter
   ```
2. **Restore dependencies:**
   ```
   dotnet restore
   ```
3. **Build the project:**
   ```
   dotnet build
   ```

---

## Usage

### Train the Model
Train the ML model with your labeled images:
```sh
# Example: train with two images labeled as having 4 and 8 rectangles

dotnet run --project AreaCounter -- train --images "Sample1.png" "Sample2.png" --labels 4 8
```
- The model will be saved as `area_counter_model.zip` in the project directory.

### Predict Rectangle Count
Predict the rectangle count for a new image:
```sh
dotnet run --project AreaCounter -- predict --image "TestImage.png"
```

### Help
For all CLI options:
```sh
dotnet run --project AreaCounter -- --help
```

---

## Project Structure

```
AreaCounter/             # Main application code
  |-- Program.cs        # CLI entry point and logic
  |-- AreaCounterService.cs  # Image processing logic (OpenCV)
  |-- MLFeatureExtractor.cs  # Feature extraction for ML
  |-- MLModelTrainer.cs      # ML.NET model training & prediction
  |-- AreaCountData.cs      # Data models for ML
  |-- ...
AreaCounter.Tests/       # xUnit test project
  |-- MLFeatureExtractorTests.cs
  |-- MLModelTrainerTests.cs
  |-- AreaCounterServiceTests.cs
  |-- AreaCountDataTests.cs
  |-- TestAssets/        # Place your test images here
```

---

## How It Works
1. **Feature Extraction:**
   - Uses OpenCV to extract contours, edge density, and aspect ratio features from input images.
2. **Model Training:**
   - ML.NET FastTree regression is trained on your labeled images and their extracted features.
3. **Prediction:**
   - For new images, features are extracted and the trained model predicts the rectangle count.

---

## Testing
- **Run all tests:**
  ```
  dotnet test AreaCounter.Tests
  ```
- Place sample PNG images in `AreaCounter.Tests/TestAssets/` for full test coverage.

---

## Extending
- Add more features to `MLFeatureExtractor.cs` to improve accuracy.
- Use more labeled images for better model generalization.
- Integrate with other ML.NET trainers or image processing techniques as needed.

---

## License
MIT License (see `LICENSE` file)

---

## Credits
- [OpenCvSharp](https://github.com/shimat/opencvsharp) for image processing
- [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet) for machine learning
- [xUnit](https://xunit.net/) for testing

---

## Contact
For questions, suggestions, or contributions, please open an issue or pull request on GitHub.
