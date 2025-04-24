using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Threading.Tasks;
using OpenCvSharp;
using System.Collections.Generic;
using ImageAreaCounter;

namespace ImageAreaCounter
{
    /// <summary>
    /// Provides core functionality to count distinct areas in an image using OpenCV.
    /// Encapsulates image processing logic.
    /// </summary>
    public static class AreaCounterService
    {
        // --- Constants ---
        private const int DefaultThresholdLightArea = 240; // Default threshold for images like Sample 1, 2 (light areas, dark lines)
        private const int DefaultThresholdDarkArea = 200;  // Default threshold for images like Sample 3 (dark areas, light lines)
        private const int MaxPixelValue = 255;             // Maximum value for an 8-bit grayscale pixel

        // Heuristic filename substring used to suggest the appropriate default threshold
        private const string DarkAreaSampleFilenameHint = "Sample 3";

        /// <summary>
        /// Analyzes an image to count distinct, externally bounded areas using contour detection.
        /// </summary>
        /// <param name="imagePath">The full path to the image file.</param>
        /// <param name="thresholdValue">The grayscale threshold value (0-255) used to binarize the image.</param>
        /// <param name="showIntermediateSteps">If true, displays the intermediate grayscale and binary images in OpenCV windows.</param>
        /// <param name="drawFinalContours">If true, displays the original image with detected contours drawn in an OpenCV window.</param>
        /// <returns>A tuple containing:
        ///   - Success (bool): True if processing completed without critical errors, false otherwise.
        ///   - Count (int): The number of distinct external areas detected, or -1 if processing failed or the image was invalid.
        /// </returns>
        /// <exception cref="FileNotFoundException">Thrown if the image file does not exist at <paramref name="imagePath"/>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if <paramref name="thresholdValue"/> is outside the valid 0-255 range.</exception>
        /// <exception cref="OpenCVException">Can be thrown by underlying OpenCV operations if image data is corrupted or other library errors occur.</exception>
        public static (bool Success, int Count) CountAreas(string imagePath, int thresholdValue, bool showIntermediateSteps = false, bool drawFinalContours = false)
        {
            // --- Input Validation ---
            ValidateInputs(imagePath, thresholdValue);

            // --- Image Processing ---
            // Use 'using' statements to ensure timely disposal of native OpenCV resources (Mat objects).
            using Mat grayImage = Cv2.ImRead(imagePath, ImreadModes.Grayscale);

            // Check if image loading failed
            if (grayImage.Empty())
            {
                Log.Error($"Failed to load or decode image: {imagePath}");
                return (Success: false, Count: -1);
            }

            // Perform thresholding to create a binary image.
            using Mat binaryImage = CreateBinaryImage(grayImage, thresholdValue);

            // Optional: Display intermediate results for debugging/visualization. Requires a GUI environment.
            if (showIntermediateSteps)
            {
                DisplayImageWindow("1. Grayscale Input", grayImage);
                DisplayImageWindow($"2. Binary (Threshold: {thresholdValue})", binaryImage);
            }

            // Find contours - the boundaries of white objects on a black background.
            // RetrievalModes.Tree finds all contours and reconstructs the full hierarchy (suitable for rectangles inside the image).
            // ContourApproximationModes.ApproxSimple compresses contour segments, saving memory.
            Cv2.FindContours(
                image: binaryImage,
                contours: out Point[][] contours,
                hierarchy: out HierarchyIndex[] hierarchy, // Hierarchy data might be useful for more complex analysis (e.g., nested shapes)
                mode: RetrievalModes.Tree,
                method: ContourApproximationModes.ApproxSimple);

            // Filter contours to count only rectangles (4-vertex, convex, area above threshold) and exclude outer border
            int rectangleCount = 0;
            for (int i = 0; i < contours.Length; i++)
            {
                // Approximate contour to polygon
                Point[] approx = Cv2.ApproxPolyDP(contours[i], 0.02 * Cv2.ArcLength(contours[i], true), true);
                // Check for 4 vertices, convex, and area threshold (to ignore noise)
                double area = Cv2.ContourArea(approx);
                if (approx.Length == 4 && Cv2.IsContourConvex(approx) && area > 1000)
                {
                    // Optionally, check if this is the outer border by seeing if it is too close to the image edge
                    Rect rect = Cv2.BoundingRect(approx);
                    const int edgeMargin = 5;
                    if (!(rect.X <= edgeMargin && rect.Y <= edgeMargin &&
                          rect.X + rect.Width >= grayImage.Width - edgeMargin &&
                          rect.Y + rect.Height >= grayImage.Height - edgeMargin))
                    {
                        rectangleCount++;
                    }
                }
            }

            // Optional: Draw detected contours on a color version of the original image. Requires a GUI environment.
            if (drawFinalContours)
            {
                DrawContoursOnImage(imagePath, contours, hierarchy);
            }

            // Return success and the rectangle count
            return (Success: true, Count: rectangleCount);
        }

        /// <summary>
        /// Validates the input parameters for the CountAreas method.
        /// </summary>
        /// <exception cref="FileNotFoundException">If image path doesn't exist.</exception>
        /// <exception cref="ArgumentOutOfRangeException">If threshold is invalid.</exception>
        private static void ValidateInputs(string imagePath, int thresholdValue)
        {
             if (string.IsNullOrWhiteSpace(imagePath)) // Added check for null/whitespace
            {
                throw new ArgumentNullException(nameof(imagePath), "Image path cannot be null or empty.");
            }
            if (!File.Exists(imagePath))
            {
                throw new FileNotFoundException($"Image file not found at the specified path: {imagePath}", imagePath);
            }
            if (thresholdValue < 0 || thresholdValue > MaxPixelValue)
            {
                 throw new ArgumentOutOfRangeException(nameof(thresholdValue), $"Threshold must be between 0 and {MaxPixelValue}.");
            }
        }

        /// <summary>
        /// Creates a binary image from a grayscale image using a specified threshold.
        /// </summary>
        /// <param name="grayImage">Input grayscale image (Mat).</param>
        /// <param name="thresholdValue">Threshold value (0-255).</param>
        /// <returns>A new binary image (Mat).</returns>
        private static Mat CreateBinaryImage(Mat grayImage, int thresholdValue)
        {
            Mat binaryImage = new Mat();
            // We consistently use BinaryInv (Inverted Binary Threshold).
            // Pixels *below* thresholdValue become white (MaxPixelValue), pixels *above* become black (0).
            // This works for both scenarios:
            // 1. Light Areas/Dark Lines: High threshold (e.g., 240). Light areas (<240) become white.
            // 2. Dark Areas/Light Lines: Low threshold (e.g., 200). Dark areas (<200) become white.
            // FindContours looks for white objects, so this ensures our target areas become white.
            const ThresholdTypes thresholdType = ThresholdTypes.BinaryInv;
            Cv2.Threshold(grayImage, binaryImage, thresholdValue, MaxPixelValue, thresholdType);
            return binaryImage;
        }


        /// <summary>
        /// Helper method to display an image in an OpenCV window. Handles potential GUI errors.
        /// </summary>
        /// <param name="windowName">The title of the OpenCV window.</param>
        /// <param name="image">The image (Mat) to display.</param>
        private static void DisplayImageWindow(string windowName, Mat image)
        {
            try
            {
                Cv2.ImShow(windowName, image);
                // WaitKey(1) allows the window to process events and render.
                // A blocking WaitKey(0) will be called later if needed.
                Cv2.WaitKey(1);
            }
            catch (Exception ex) // Catch potential errors if no GUI is available
            {
                Log.Warning($"Could not display window '{windowName}': {ex.Message}. Ensure a GUI environment is available for visualization.");
            }
        }

         /// <summary>
        /// Helper method to draw detected contours onto a color version of the original image for visualization.
        /// </summary>
        /// <param name="originalImagePath">Path to the original image file (to load in color).</param>
        /// <param name="contours">The array of contours (Point arrays) detected.</param>
        /// <param name="hierarchy">The hierarchy information associated with the contours.</param>
        private static void DrawContoursOnImage(string originalImagePath, Point[][] contours, HierarchyIndex[] hierarchy)
        {
            // Load the original image in color to draw upon.
            using Mat colorImage = Cv2.ImRead(originalImagePath, ImreadModes.Color);
            if (colorImage.Empty())
            {
                Log.Warning("Could not reload the original image in color for drawing contours.");
                return;
            }

            Log.Info("Drawing detected contours on image...");
            for (int i = 0; i < contours.Length; i++)
            {
                // Assign a unique, random color to each contour for better visibility.
                Scalar color = Scalar.RandomColor();
                // Draw the i-th contour. Thickness 2 makes it clearly visible.
                Cv2.DrawContours(colorImage, contours, i, color, thickness: 2, lineType: LineTypes.Link8, hierarchy: hierarchy);
            }

            // Display the image with contours.
            DisplayImageWindow("3. Detected Contours", colorImage);
            Log.Info("Press any key in the 'Detected Contours' window to close it.");
            Cv2.WaitKey(0); // Wait indefinitely until the user presses a key *in the OpenCV window*.
            Cv2.DestroyWindow("3. Detected Contours"); // Explicitly destroy the window.
        }


        /// <summary>
        /// Attempts to close all open OpenCV windows. Safe to call even if no windows are open.
        /// </summary>
        public static void DestroyAllWindows()
        {
            try
            {
                Cv2.DestroyAllWindows();
            }
            catch (Exception ex)
            {
                // Log lightly, as this can happen if no windows were ever created (e.g., in headless mode)
                Log.Debug($"Exception during DestroyAllWindows (may be normal if no GUI): {ex.Message}");
            }
        }

        /// <summary>
        /// Suggests a default threshold value based on simple filename heuristics.
        /// </summary>
        /// <param name="imagePath">Path to the image file.</param>
        /// <returns>Suggested threshold value (either DefaultThresholdDarkArea or DefaultThresholdLightArea).</returns>
        public static int SuggestThreshold(string imagePath)
        {
            string fileName = Path.GetFileName(imagePath);
            // If the filename contains our hint for dark-area images, suggest the lower threshold.
            if (!string.IsNullOrEmpty(fileName) && fileName.Contains(DarkAreaSampleFilenameHint, StringComparison.OrdinalIgnoreCase))
            {
                Log.Info($"Filename contains '{DarkAreaSampleFilenameHint}'. Suggesting lower threshold ({DefaultThresholdDarkArea}) for potentially dark areas.");
                return DefaultThresholdDarkArea;
            }
            // Otherwise, assume light areas and suggest the higher threshold.
            Log.Info($"Suggesting higher threshold ({DefaultThresholdLightArea}) for potentially light areas.");
            return DefaultThresholdLightArea;
        }
    }

    /// <summary>
    /// Main program class responsible for parsing command-line arguments,
    /// orchestrating the area counting process, and handling console output.
    /// </summary>
    public class Program
    {
        /// <summary>
        /// Application entry point. Configures and invokes the command-line parser.
        /// </summary>
        /// <param name="args">Command-line arguments passed to the application.</param>
        /// <returns>Exit code (0 for success, 1 for failure).</returns>
        public static async Task<int> Main(string[] args)
        {
            // --- Command Line Argument Setup (using System.CommandLine) ---

            // Define the required argument for the image file path.
            var fileArgument = new Argument<FileInfo>(
                name: "image-path", // Name used in help text and parsing
                description: "Path to the input image file.")
            {
                // Add validation: Argument must represent an existing file.
                Arity = ArgumentArity.ExactlyOne // Explicitly state that one value is required
            };
            fileArgument.ExistingOnly(); // Built-in validation

            // Define the optional threshold value.
            var thresholdOption = new Option<int?>( // Use nullable int (int?) to detect if the user provided it
                name: "--threshold", // Option names typically start with --
                description: "Grayscale threshold value (0-255). If omitted, a default is suggested based on filename.");

            // Define the optional flag to show intermediate processing steps.
            var showStepsOption = new Option<bool>(
                name: "--show-steps",
                description: "Display intermediate grayscale and binary images (requires GUI).",
                getDefaultValue: () => false); // Default is false if the flag isn't present

             // Define the optional flag to draw final contours.
             var drawContoursOption = new Option<bool>(
                name: "--draw-contours",
                description: "Display the final image with detected contours highlighted (requires GUI).",
                getDefaultValue: () => false); // Default is false

            // Configure the root command for the application.
            var rootCommand = new RootCommand("Counts distinct areas in an image using OpenCV contour detection.")
            {
                // Add the argument and options to the command.
                fileArgument,
                thresholdOption,
                showStepsOption,
                drawContoursOption
            };

            // Set the handler that will be executed when the command is invoked.
            // This lambda receives the parsed values for the arguments and options.
            rootCommand.SetHandler(async (InvocationContext context) =>
            {
                // Retrieve the parsed values from the invocation context.
                var imageFile = context.ParseResult.GetValueForArgument(fileArgument);
                var threshold = context.ParseResult.GetValueForOption(thresholdOption); // Will be null if not provided
                var showSteps = context.ParseResult.GetValueForOption(showStepsOption);
                var drawContours = context.ParseResult.GetValueForOption(drawContoursOption);

                // Call the main analysis logic and set the exit code based on its result.
                context.ExitCode = await RunAnalysisLogic(imageFile, threshold, showSteps, drawContours);
            });

            // --- Execute Command ---
            // Invoke the command-line parser with the provided arguments.
            return await rootCommand.InvokeAsync(args);
        }

        /// <summary>
        /// Orchestrates the image analysis process based on parsed command-line arguments.
        /// Handles overall execution flow, logging, and error management.
        /// </summary>
        /// <param name="imageFile">FileInfo object representing the validated input image file.</param>
        /// <param name="threshold">The user-provided threshold (nullable int), or null if not provided.</param>
        /// <param name="showSteps">Flag indicating whether to show intermediate steps.</param>
        /// <param name="drawContours">Flag indicating whether to draw final contours.</param>
        /// <returns>Exit code (0 for success, 1 for failure).</returns>
        private static async Task<int> RunAnalysisLogic(FileInfo imageFile, int? threshold, bool showSteps, bool drawContours)
        {
            Log.Info("--- Image Area Counter ---", ConsoleColor.Cyan);
            Log.Info($"Processing image: {imageFile.FullName}");

            int effectiveThreshold;
            // Determine the threshold to use: user-provided or suggested.
            if (threshold.HasValue)
            {
                effectiveThreshold = threshold.Value;
                Log.Info($"Using provided threshold: {effectiveThreshold}");
            }
            else
            {
                // Suggest a threshold if none was provided by the user.
                effectiveThreshold = AreaCounterService.SuggestThreshold(imageFile.FullName);
                // Log.Info($"Using suggested threshold: {effectiveThreshold}"); // SuggestThreshold already logs this
            }

            int finalCount = -1;
            bool success = false;
            int exitCode = 1; // Default to failure exit code

            try
            {
                Log.Info("Starting analysis...");
                // Call the core processing logic from the service class.
                (success, finalCount) = AreaCounterService.CountAreas(
                    imageFile.FullName,
                    effectiveThreshold,
                    showSteps,
                    drawContours);

                // Report the final result based on success status.
                if (success)
                {
                    Log.Success($"\n>>> Analysis Complete: Detected {finalCount} areas. <<<");
                    exitCode = 0; // Set success exit code
                }
                else
                {
                    // CountAreas should have logged specific errors if possible.
                    Log.Warning("\nAnalysis finished, but reported an issue during processing.");
                }
            }
            // Catch specific, expected exceptions for cleaner error reporting.
            catch (FileNotFoundException fnfEx)
            {
                Log.Error($"\nFile Error: {fnfEx.Message}");
            }
            catch (ArgumentException argEx) // Catches ArgumentOutOfRangeException too
            {
                Log.Error($"\nArgument Error: {argEx.Message}");
            }
            // Catch broader OpenCV exceptions.
            catch (OpenCVException cvEx)
            {
                 Log.Error($"\nOpenCV Error: {cvEx.Message}");
                 Log.Debug($"OpenCV StackTrace: {cvEx.StackTrace}"); // Debug log for more detail
            }
            // Catch any other unexpected exceptions.
            catch (Exception ex)
            {
                Log.Error($"\nAn unexpected error occurred: {ex.Message}");
                // Log the full exception details for debugging, possibly to a file in a real application.
                Log.Debug($"StackTrace: {ex.ToString()}");
                // Environment.FailFast("Critical unexpected error", ex); // Consider if failure should be immediate
            }
            finally
            {
                 // Ensure OpenCV windows are closed if they were potentially opened.
                 if (showSteps || drawContours)
                 {
                     Log.Info("Press any key in this console window to close all OpenCV windows and exit...");
                     Console.ReadKey(); // Wait for user acknowledgment before windows disappear.
                     AreaCounterService.DestroyAllWindows();
                 }
                 else
                 {
                      Log.Info("\nProcessing finished.");
                 }
            }

            return exitCode; // Return 0 for success, 1 for failure
        }
    }

    /// <summary>
    /// Simple static logger class for consistent console output formatting.
    /// </summary>
    internal static class Log
    {
        public static void Info(string message, ConsoleColor color = ConsoleColor.Gray) => Write(message, color);
        public static void Success(string message, ConsoleColor color = ConsoleColor.Green) => Write(message, color);
        public static void Warning(string message, ConsoleColor color = ConsoleColor.Yellow) => Write(message, color, Console.Error);
        public static void Error(string message, ConsoleColor color = ConsoleColor.Red) => Write(message, color, Console.Error);
        public static void Debug(string message, ConsoleColor color = ConsoleColor.DarkGray)
        {
#if DEBUG
            Write($"DEBUG: {message}", color, Console.Error);
#endif
        }
        private static void Write(string message, ConsoleColor color, TextWriter? target = null)
        {
            Console.ForegroundColor = color;
            (target ?? Console.Out).WriteLine(message);
            Console.ResetColor();
        }
    }

    // --- ML.NET CLI ---
    /// <summary>
    /// Provides the general command-line interface for the AreaCounter application.
    /// Handles training and prediction commands for the rectangle counting model.
    /// </summary>
    public static class AreaCounterCLI
    {
        private const string ModelPath = "area_counter_model.zip";

        /// <summary>
        /// Entry point for the CLI. Configures and invokes the command-line parser.
        /// </summary>
        /// <param name="args">Command-line arguments.</param>
        public static void Run(string[] args)
        {
            var rootCmd = new RootCommand("AreaCounter CLI");

            // --- Train Command ---
            var imagesOption = new Option<List<string>>(
                name: "--images",
                description: "List of image paths for training") { IsRequired = true };
            var labelsOption = new Option<List<float>>(
                name: "--labels",
                description: "Number of rectangles in each image") { IsRequired = true };
            var trainCmd = new Command("train", "Train the ML model with labeled images");
            trainCmd.AddOption(imagesOption);
            trainCmd.AddOption(labelsOption);
            trainCmd.SetHandler(async (List<string> images, List<float> labels) =>
            {
                if (images.Count != labels.Count)
                {
                    Log.Error("The number of images and labels must match.");
                    return;
                }
                var trainingData = images.Zip(labels, (img, label) => (img, label)).ToList();
                MLModelTrainer.TrainAndSaveModel(trainingData, ModelPath);
                Log.Success($"Model trained and saved to {ModelPath}");
                await Task.CompletedTask;
            }, imagesOption, labelsOption);
            rootCmd.AddCommand(trainCmd);

            // --- Predict Command ---
            var imageOption = new Option<string>(
                name: "--image",
                description: "Image path to predict") { IsRequired = true };
            var predictCmd = new Command("predict", "Predict rectangle count for an image");
            predictCmd.AddOption(imageOption);
            predictCmd.SetHandler(async (string image) =>
            {
                float pred = MLModelTrainer.PredictAreaCount(image, ModelPath);
                Log.Success($"Predicted rectangle count: {Math.Round(pred)} (raw: {pred:F2})");
                await Task.CompletedTask;
            }, imageOption);
            rootCmd.AddCommand(predictCmd);

            rootCmd.Invoke(args);
        }
    }

    // --- Entry Point ---

}
