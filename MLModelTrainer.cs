using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;

namespace ImageAreaCounter
{
    /// <summary>
    /// Provides methods to train and use an ML.NET regression model for rectangle counting.
    /// </summary>
    public static class MLModelTrainer
    {
        /// <summary>
        /// Trains a FastTree regression model using extracted features and saves it to disk.
        /// </summary>
        /// <param name="trainingData">A list of tuples (imagePath, areaCount) with ground truth labels.</param>
        /// <param name="modelPath">The file path to save the trained model.</param>
        public static void TrainAndSaveModel(List<(string imagePath, float areaCount)> trainingData, string modelPath)
        {
            if (trainingData == null || trainingData.Count == 0)
                throw new ArgumentException("Training data cannot be null or empty.", nameof(trainingData));

            var mlContext = new MLContext();
            var featureData = new List<AreaCountData>();
            foreach (var (imagePath, areaCount) in trainingData)
            {
                var features = MLFeatureExtractor.ExtractFeatures(imagePath);
                featureData.Add(new AreaCountData
                {
                    ContourCount = features.ContourCount,
                    AvgContourArea = features.AvgContourArea,
                    StdContourArea = features.StdContourArea,
                    EdgeDensity = features.EdgeDensity,
                    AvgRectAspectRatio = features.AvgRectAspectRatio,
                    AreaCount = areaCount
                });
            }
            var dataView = mlContext.Data.LoadFromEnumerable(featureData);
            var pipeline = mlContext.Transforms.Concatenate("Features",
                        nameof(AreaCountData.ContourCount),
                        nameof(AreaCountData.AvgContourArea),
                        nameof(AreaCountData.StdContourArea),
                        nameof(AreaCountData.EdgeDensity),
                        nameof(AreaCountData.AvgRectAspectRatio))
                .Append(mlContext.Regression.Trainers.FastTree());
            var model = pipeline.Fit(dataView);
            mlContext.Model.Save(model, dataView.Schema, modelPath);
        }

        /// <summary>
        /// Loads a trained model and predicts the rectangle count for a new image.
        /// </summary>
        /// <param name="imagePath">Path to the image to predict.</param>
        /// <param name="modelPath">Path to the trained model file.</param>
        /// <returns>Predicted rectangle count (float).</returns>
        public static float PredictAreaCount(string imagePath, string modelPath)
        {
            var mlContext = new MLContext();
            var features = MLFeatureExtractor.ExtractFeatures(imagePath);
            var predEngine = mlContext.Model.CreatePredictionEngine<AreaCountData, AreaCountPrediction>(
                mlContext.Model.Load(modelPath, out _));
            var input = new AreaCountData
            {
                ContourCount = features.ContourCount,
                AvgContourArea = features.AvgContourArea,
                StdContourArea = features.StdContourArea,
                EdgeDensity = features.EdgeDensity,
                AvgRectAspectRatio = features.AvgRectAspectRatio,
            };
            var prediction = predEngine.Predict(input);
            return prediction.AreaCount;
        }
    }
}
