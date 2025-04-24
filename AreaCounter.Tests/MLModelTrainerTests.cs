using System;
using System.Collections.Generic;
using Xunit;
using ImageAreaCounter;
using System.IO;

namespace AreaCounter.Tests
{
    public class MLModelTrainerTests
    {
        [Fact]
        public void TrainAndSaveModel_ThrowsOnEmptyData()
        {
            Assert.Throws<ArgumentException>(() =>
                MLModelTrainer.TrainAndSaveModel(new List<(string, float)>(), "dummy_model.zip"));
        }

        [Fact]
        public void TrainAndPredict_WorksWithValidData()
        {
            // Arrange
            string testImage = Path.Combine("TestAssets", "rectangles_2.png");
            if (!File.Exists(testImage))
                throw new SkipTestException($"Test image not found: {testImage}");

            string modelPath = "test_model.zip";
            var data = new List<(string, float)> { (testImage, 2) };

            // Act
            MLModelTrainer.TrainAndSaveModel(data, modelPath);
            float prediction = MLModelTrainer.PredictAreaCount(testImage, modelPath);

            // Assert
            Assert.True(prediction >= 0);
            File.Delete(modelPath);
        }
    }
}
