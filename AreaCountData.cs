using Microsoft.ML.Data;

namespace ImageAreaCounter
{
    /// <summary>
    /// ML.NET data model for rectangle counting. Holds extracted features and ground truth label.
    /// </summary>
    public class AreaCountData
    {
        /// <summary>Number of detected contours in the image.</summary>
        [LoadColumn(0)] public float ContourCount { get; set; }
        /// <summary>Average area of detected contours.</summary>
        [LoadColumn(1)] public float AvgContourArea { get; set; }
        /// <summary>Standard deviation of contour areas.</summary>
        [LoadColumn(2)] public float StdContourArea { get; set; }
        /// <summary>Density of edges in the image (fraction of edge pixels).</summary>
        [LoadColumn(3)] public float EdgeDensity { get; set; }
        /// <summary>Average aspect ratio of bounding rectangles.</summary>
        [LoadColumn(4)] public float AvgRectAspectRatio { get; set; }
        /// <summary>Label: actual rectangle count (for training).</summary>
        [LoadColumn(5)] public float AreaCount { get; set; }
    }

    /// <summary>
    /// ML.NET prediction result for rectangle count.
    /// </summary>
    public class AreaCountPrediction
    {
        /// <summary>Predicted rectangle count (output of regression model).</summary>
        [ColumnName("Score")] public float AreaCount;
    }
}
