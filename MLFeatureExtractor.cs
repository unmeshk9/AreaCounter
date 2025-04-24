using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;

namespace ImageAreaCounter
{
    /// <summary>
    /// Represents extracted features from an image for ML input.
    /// </summary>
    public class ImageFeatures
    {
        public float ContourCount { get; set; }
        public float AvgContourArea { get; set; }
        public float StdContourArea { get; set; }
        public float EdgeDensity { get; set; }
        public float AvgRectAspectRatio { get; set; }
    }

    /// <summary>
    /// Provides methods to extract ML-relevant features from images using OpenCV.
    /// </summary>
    public static class MLFeatureExtractor
    {
        /// <summary>
        /// Extracts features (contour count, area stats, edge density, aspect ratio) from an image.
        /// </summary>
        /// <param name="imagePath">Path to the image file.</param>
        /// <returns>ImageFeatures object with extracted values.</returns>
        public static ImageFeatures ExtractFeatures(string imagePath)
        {
            using var src = Cv2.ImRead(imagePath, ImreadModes.Grayscale);
            if (src.Empty())
                throw new FileNotFoundException($"Image not found: {imagePath}");

            // 1. Canny Edge Detection for edge density
            using var edges = new Mat();
            Cv2.Canny(src, edges, 100, 200);
            float edgeDensity = (float)Cv2.CountNonZero(edges) / (src.Rows * src.Cols);

            // 2. Binary threshold for contour detection
            using var binary = new Mat();
            Cv2.Threshold(src, binary, 200, 255, ThresholdTypes.BinaryInv);

            // 3. Contour Detection
            Cv2.FindContours(binary, out Point[][] contours, out _, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);
            float contourCount = contours.Length;
            var areas = new List<float>();
            var aspectRatios = new List<float>();
            foreach (var contour in contours)
            {
                double area = Cv2.ContourArea(contour);
                if (area > 100) // Ignore tiny noise
                {
                    areas.Add((float)area);
                    var rect = Cv2.BoundingRect(contour);
                    aspectRatios.Add(rect.Height > 0 ? (float)rect.Width / rect.Height : 0f);
                }
            }
            float avgArea = areas.Count > 0 ? areas.Average() : 0f;
            float stdArea = areas.Count > 1 ? (float)Math.Sqrt(areas.Select(a => Math.Pow(a - avgArea, 2)).Average()) : 0f;
            float avgAspect = aspectRatios.Count > 0 ? aspectRatios.Average() : 0f;

            return new ImageFeatures
            {
                ContourCount = contourCount,
                AvgContourArea = avgArea,
                StdContourArea = stdArea,
                EdgeDensity = edgeDensity,
                AvgRectAspectRatio = avgAspect
            };
        }
    }
}
