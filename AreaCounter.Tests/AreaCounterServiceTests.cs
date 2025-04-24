using System;
using Xunit;
using ImageAreaCounter;
using System.IO;

namespace AreaCounter.Tests
{
    public class AreaCounterServiceTests
    {
        [Fact]
        public void CountAreas_ThrowsOnInvalidThreshold()
        {
            string testImage = Path.Combine("TestAssets", "rectangles_2.png");
            if (!File.Exists(testImage))
                throw new SkipTestException($"Test image not found: {testImage}");

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                AreaCounterService.CountAreas(testImage, -1));
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                AreaCounterService.CountAreas(testImage, 999));
        }

        [Fact]
        public void CountAreas_ReturnsResultForValidImage()
        {
            string testImage = Path.Combine("TestAssets", "rectangles_2.png");
            if (!File.Exists(testImage))
                throw new SkipTestException($"Test image not found: {testImage}");

            var (success, count) = AreaCounterService.CountAreas(testImage, 200);
            Assert.True(success);
            Assert.True(count >= 0);
        }
    }
}
