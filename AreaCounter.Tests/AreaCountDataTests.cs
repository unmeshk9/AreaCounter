using Xunit;
using ImageAreaCounter;

namespace AreaCounter.Tests
{
    public class AreaCountDataTests
    {
        [Fact]
        public void AreaCountData_Defaults_AreZero()
        {
            var data = new AreaCountData();
            Assert.Equal(0, data.ContourCount);
            Assert.Equal(0, data.AvgContourArea);
            Assert.Equal(0, data.StdContourArea);
            Assert.Equal(0, data.EdgeDensity);
            Assert.Equal(0, data.AvgRectAspectRatio);
            Assert.Equal(0, data.AreaCount);
        }

        [Fact]
        public void AreaCountPrediction_Default_IsZero()
        {
            var pred = new AreaCountPrediction();
            Assert.Equal(0, pred.AreaCount);
        }
    }
}
