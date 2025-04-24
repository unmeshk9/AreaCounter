using System;
using Xunit;
using ImageAreaCounter;
using System.IO;

namespace AreaCounter.Tests
{
    public class MLFeatureExtractorTests
    {
        [Fact]
        public void ExtractFeatures_ThrowsIfImageNotFound()
        {
            Assert.Throws<FileNotFoundException>(() =>
                MLFeatureExtractor.ExtractFeatures("nonexistent_image.png"));
        }

        [Fact]
        public void ExtractFeatures_ReturnsFeaturesForValidImage()
        {
            // Arrange: Use a known test image (should be present in test assets)
            string testImage = Path.Combine("TestAssets", "rectangles_2.png");
            if (!File.Exists(testImage))
                throw new SkipTestException($"Test image not found: {testImage}");

            // Act
            var features = MLFeatureExtractor.ExtractFeatures(testImage);

            // Assert - values will depend on the image, just check plausible ranges
            Assert.True(features.ContourCount >= 0);
            Assert.True(features.EdgeDensity >= 0 && features.EdgeDensity <= 1);
        }
    }

    public class SkipTestException : Exception
    {
        public SkipTestException(string message) : base(message) { }
    }
}
