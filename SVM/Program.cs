using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace SVM
{
    class Program
    {
        private static List<List<double>> features;
        private static List<int> classValues;
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            features = new List<List<double>>();
            classValues = new List<int>();
            ReadData(out double[] min, out double[] max);
            NormalizeData(min, max);

            Console.WriteLine("Training data:");
            for (int i = 0; i < features.Count; ++i)
            {
                Console.Write("[" + i + "] ");
                for (int j = 0; j < features[i].Count; ++j)
                    Console.Write(features[i][j].ToString("F6").PadLeft(10));
                Console.WriteLine("  |  " + classValues[i].ToString().PadLeft(3));
            }

            double[] possibleCoefs = new[] {0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0};
            int[] possibleDegrees = new[] {1, 2, 3, 4, 5};

            
            int maxIter = (int)  1e4;

            bool poly = false; //set poly or rbf

            foreach (var coef in possibleCoefs)
            {
                if (poly)
                {
                    foreach (var degree in possibleDegrees)
                    {
                        Console.WriteLine($"{coef} coef and {degree} degree");
                        var svm = new SupportVectorMachine("poly", 0); // poly kernel, seed
                        SupportVectorMachine.Degree = degree;
                        SupportVectorMachine.Coef = coef;
                        Console.WriteLine("Starting training");
                        int iter = svm.Train(features, classValues, maxIter);
                        Console.WriteLine("Training complete in " + iter + " iterations\n");

                        Console.WriteLine($"Support vectors ({svm.SupportVectors.Count}):");
                        /*foreach (var vec in svm.supportVectors)
                        {
                            for (int i = 0; i < vec.Count; ++i)
                                Console.Write(vec[i].ToString("F6") + "  ");
                            Console.WriteLine("");
                        }
    
                        Console.WriteLine("\nWeights: ");
                        foreach (var t in svm.Weights)
                        {
                            Console.Write(t.ToString("F6") + " ");
                        }*/

                        Console.WriteLine("");

                        Console.WriteLine("\nBias = " + svm.Bias.ToString("F6") + "\n");

                        for (int i = 0; i < features.Count; ++i)
                        {
                            double pred = svm.ComputeDecision(features[i]);
                            //Console.Write("Predicted decision value for [" + i + "] = ");
                            //Console.WriteLine(pred.ToString("F6").PadLeft(10));
                        }

                        double acc = svm.Accuracy(features, classValues);
                        Console.WriteLine("\nModel accuracy on test data = " +
                                          acc.ToString("F6"));
                    }
                }
                else
                {
                    for (double gamma = 1; gamma < 5.0; gamma += 0.05)
                    {
                        Console.WriteLine($"{coef} coef and {gamma} gamma");
                        var svm = new SupportVectorMachine("poly", 0); // poly kernel, seed
                        SupportVectorMachine.Gamma = gamma;
                        SupportVectorMachine.Coef = coef;
                        Console.WriteLine("Starting training");
                        int iter = svm.Train(features, classValues, maxIter);
                        Console.WriteLine("Training complete in " + iter + " iterations\n");

                        Console.WriteLine($"Support vectors ({svm.SupportVectors.Count}):");
                        /*foreach (var vec in svm.supportVectors)
                        {
                            for (int i = 0; i < vec.Count; ++i)
                                Console.Write(vec[i].ToString("F6") + "  ");
                            Console.WriteLine("");
                        }
    
                        Console.WriteLine("\nWeights: ");
                        foreach (var t in svm.Weights)
                        {
                            Console.Write(t.ToString("F6") + " ");
                        }*/

                        Console.WriteLine("");

                        Console.WriteLine("\nBias = " + svm.Bias.ToString("F6") + "\n");

                        for (int i = 0; i < features.Count; ++i)
                        {
                            double pred = svm.ComputeDecision(features[i]);
                            //Console.Write("Predicted decision value for [" + i + "] = ");
                            //Console.WriteLine(pred.ToString("F6").PadLeft(10));
                        }

                        double acc = svm.Accuracy(features, classValues);
                        Console.WriteLine("\nModel accuracy on test data = " +
                                          acc.ToString("F6"));
                    }
                }
            }
            
        }

        private static void ReadData(out double[] min, out double[] max)
        {
            max = new[] { 0.0, 0.0 };
            min = new[] { Double.MaxValue, Double.MaxValue,};
            Dictionary<string, int> classes = new Dictionary<string, int>()
            {
                {"N", -1},
                {"P", 1}
            };
            using (var reader = new StreamReader(@"..\..\..\..\geyser.csv"))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line?.Split(',');
                    if (Double.TryParse(values?[0], System.Globalization.NumberStyles.Number, new CultureInfo("en-US"),  out _))
                    {
                        features.Add(new List<double>());
                    }
                    else
                    {
                        continue;
                    }

                    for (int i = 0; i < values?.Length - 1; i++)
                    {
                        Double.TryParse(values[i], System.Globalization.NumberStyles.Number, new CultureInfo("en-US"), out var val);
                        features.Last().Add(val);
                        min[i] = Math.Min(min[i], val);
                        max[i] = Math.Max(max[i], val);
                    }

                    classValues.Add(classes[values[^1]]);
                }
            }
        }

        private static void NormalizeData(double[] min, double[] max)
        {
            foreach (var dataString in features)
            {
                for (int i = 0; i < dataString.Count; i++)
                {
                    dataString[i] -= min[i];
                    dataString[i] /= (max[i] - min[i]);
                }
            }
        }
    }
    public class SupportVectorMachine
    {
        public Random Rnd;

        public double Complexity = 1.0;
        public double Tolerance = 1.0e-9;  // error tolerance
        public double Epsilon = 1.0e-9;

        public List<List<double>> SupportVectors;  // at least 2 of them
        public double[] Weights;  // one weight per support vector
        public double[] Alpha;    // one alpha per training item
        public double Bias;

        public double[] Errors;  // cache. one per training item

        public static double Gamma = 1;
        public static double Coef = 1;
        public static int Degree = 5;

        public Func<List<double>, List<double>, double> KernelFunc = RbfKernel;

        public SupportVectorMachine(string kernelType, int seed)
        {
            if (kernelType != "poly")
                throw new Exception("This SVM uses hard-coded polynomial kernel");
            this.Rnd = new Random(seed);
            this.SupportVectors = new List<List<double>>();
            // this.weights allocated after know how many support vecs there are
        }

        public static double LinearKernel(List<double> v1, List<double> v2)
        {
            return v1.Select((t, i) => t * v2[i]).Sum() + Coef;
        }

        public static double PolyKernel(List<double> v1, List<double> v2)
        {
            var sum = v1.Select((t, i) => t * v2[i]).Sum();
            double z = sum + Coef;
            return Math.Pow(z, Degree);
        }

        public static double RbfKernel(List<double> v1, List<double> v2)
        {
            double sum = 0.0;
            for (int i = 0; i < v1.Count; ++i)
                sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
            return Math.Exp(-Gamma * sum);
        }

        public double ComputeDecision(List<double> input)
        {
            double sum = this.SupportVectors.Select((t, i) => this.Weights[i] * KernelFunc(t, input)).Sum();
            sum += this.Bias;
            return sum;
        }
        public int Train(List<List<double>> features, List<int> classValues, int maxIter)
        {
            int n = features.Count;
            this.Alpha = new double[n];
            this.Errors = new double[n];
            int numChanged = 0;
            bool examineAll = true;
            int iter = 0;

            while (iter < maxIter && numChanged > 0 || examineAll)
            {
                ++iter;
                numChanged = 0;
                if (examineAll)
                {
                    // all training examples
                    for (int i = 0; i < n; i++)
                        numChanged += ExamineExample(i, features, classValues);
                }
                else
                {
                    // examples where alpha is not 0 and not C
                    for (int i = 0; i < n; i++)
                        if (this.Alpha[i] != 0 && this.Alpha[i] != this.Complexity)
                            numChanged += ExamineExample(i, features, classValues);
                }

                if (examineAll)
                    examineAll = false;
                else if (numChanged == 0)
                    examineAll = true;
            }

            List<int> indices = new List<int>();  // indices of support vectors
            for (int i = 0; i < n; i++)
            {
                // Only store vectors with Lagrange multipliers > 0
                if (this.Alpha[i] > 0) indices.Add(i);
            }

            int numSuppVectors = indices.Count;
            this.Weights = new double[numSuppVectors];
            for (int i = 0; i < numSuppVectors; i++)
            {
                int j = indices[i];
                this.SupportVectors.Add(features[j]);
                this.Weights[i] = this.Alpha[j] * classValues[j];
            }
            this.Bias = -1 * this.Bias;
            return iter;
        }
        
        public double Accuracy(List<List<double>> features, List<int> classValues)
        {
            // Compute classification accuracy
            int numCorrect = 0; int numWrong = 0;
            for (int i = 0; i < features.Count; ++i)
            {
                double signComputed = Math.Sign(ComputeDecision(features[i]));
                if (signComputed == Math.Sign(classValues[i]))
                    ++numCorrect;
                else
                    ++numWrong;
            }

            return (1.0 * numCorrect) / (numCorrect + numWrong);
        }

        
        private bool TakeStep(int i1, int i2,
          List<List<double>> features, List<int> classValues)
        {
            // "Sequential Minimal Optimization
            if (i1 == i2) return false;

            double C = this.Complexity;
            double eps = this.Epsilon;

            List<double> x1 = new List<double>(features[i1]);  // "point" at index i1
            double alph1 = this.Alpha[i1];    // Lagrange multiplier for i1
            double y1 = classValues[i1];    // label

            double E1;
            if (alph1 > 0 && alph1 < C)
                E1 = this.Errors[i1];
            else
                E1 = ComputeAll(x1, features, classValues) - y1;

            List<double> x2 = new List<double>(features[i2]);  // index i2
            double alph2 = this.Alpha[i2];
            double y2 = classValues[i2];

            // SVM output on point [i2] - y2 (check in error cache)
            double E2;
            if (alph2 > 0 && alph2 < C)
                E2 = this.Errors[i2];
            else
                E2 = ComputeAll(x2, features, classValues) - y2;

            double s = y1 * y2;

            // Compute L and H via equations (13) and (14)
            double L, H;
            if (y1 != y2)
            {
                L = Math.Max(0, alph2 - alph1);  // 13a
                H = Math.Min(C, C + alph2 - alph1);  // 13b
            }
            else
            {
                L = Math.Max(0, alph2 + alph1 - C);  // 14a
                H = Math.Min(C, alph2 + alph1);  // 14b
            }

            if (L == H) return false;

            double k11 = KernelFunc(x1, x1);  // conveniences
            double k12 = KernelFunc(x1, x2);
            double k22 = KernelFunc(x2, x2);
            double eta = k11 + k22 - 2 * k12;  // 15

            double a1, a2;
            if (eta > 0)
            {
                a2 = alph2 - y2 * (E2 - E1) / eta;  // 16

                if (a2 >= H) a2 = H;  // 17a
                else if (a2 <= L) a2 = L;  // 17b
            }
            else  // "Under unusual circumstances, eta will not be positive"
            {
                double f1 =
                  y1 * (E1 + this.Bias) - alph1 * k11 - s * alph2 * k12;  // 19a
                double f2 =
                  y2 * (E2 + this.Bias) - alph2 * k22 - s * alph1 * k12;  // 19b
                double L1 = alph1 + s * (alph2 - L);  // 19c
                double H1 = alph1 + s * (alph2 - H);  // 19d
                double Lobj = (L1 * f1) + (L * f2) + (0.5 * L1 * L1 * k11) +
                  (0.5 * L * L * k22) + (s * L * L1 * k12);  // 19e
                double Hobj = (H1 * f1) + (H * f2) + (0.5 * H1 * H1 * k11) +
                  (0.5 * H * H * k22) + (s * H * H1 * k12);  // 19f

                if (Lobj < Hobj - eps) a2 = L;
                else if (Lobj > Hobj + eps) a2 = H;
                else a2 = alph2;
            }

            if (Math.Abs(a2 - alph2) < eps * (a2 + alph2 + eps))
                return false;

            a1 = alph1 + s * (alph2 - a2);  // 18

            // Update threshold (biasa). See section 2.3
            double b1 = E1 + y1 * (a1 - alph1) * k11 +
              y2 * (a2 - alph2) * k12 + this.Bias;
            double b2 = E2 + y1 * (a1 - alph1) * k12 +
              y2 * (a2 - alph2) * k22 + this.Bias;
            double newb;
            if (0 < a1 && C > a1)
                newb = b1;
            else if (0 < a2 && C > a2)
                newb = b2;
            else
                newb = (b1 + b2) / 2;

            double deltab = newb - this.Bias;
            this.Bias = newb;

            // Update error cache using new Lagrange multipliers
            double delta1 = y1 * (a1 - alph1);
            double delta2 = y2 * (a2 - alph2);

            for (int i = 0; i < features.Count; ++i)
            {
                if (0 < this.Alpha[i] && this.Alpha[i] < C)
                    this.Errors[i] += delta1 *
                      KernelFunc(x1, features[i]) +
                      delta2 * KernelFunc(x2, features[i]) - deltab;
            }

            this.Errors[i1] = 0.0;
            this.Errors[i2] = 0.0;
            this.Alpha[i1] = a1;
            this.Alpha[i2] = a2;

            return true;
        }

        private int ExamineExample(int i2, List<List<double>> features, List<int> classValues)
        {
            double C = this.Complexity;
            double tol = this.Tolerance;

            List<double> x2 = new List<Double>(features[i2]); // "point" at i2
            double y2 = classValues[i2];   // class label for p2
            double alph2 = this.Alpha[i2];   // Lagrange multiplier for i2

            // SVM output on point[i2] - y2. (check in error cache)
            double E2;
            if (alph2 > 0 && alph2 < C)
                E2 = this.Errors[i2];
            else
                E2 = ComputeAll(x2, features, classValues) - y2;

            double r2 = y2 * E2;

            if ((r2 < -tol && alph2 < C) || (r2 > tol && alph2 > 0))
            {
                // See section 2.2
                int i1 = -1; 
                double maxErr = 0;
                for (int i = 0; i < features.Count; ++i)
                {
                    if (this.Alpha[i] > 0 && this.Alpha[i] < C)
                    {
                        double E1 = this.Errors[i];
                        double delErr = System.Math.Abs(E2 - E1);

                        if (delErr > maxErr)
                        {
                            maxErr = delErr;
                            i1 = i;
                        }
                    }
                }

                if (i1 >= 0 && TakeStep(i1, i2, features, classValues)) return 1;

                int rndi = this.Rnd.Next(features.Count);
                for (i1 = rndi; i1 < features.Count; ++i1)
                {
                    if (this.Alpha[i1] > 0 && this.Alpha[i1] < C)
                        if (TakeStep(i1, i2, features, classValues)) return 1;
                }
                for (i1 = 0; i1 < rndi; ++i1)
                {
                    if (this.Alpha[i1] > 0 && this.Alpha[i1] < C)
                        if (TakeStep(i1, i2, features, classValues)) return 1;
                }

                // "Both the iteration through the non-bound examples and the
                // iteration through the entire training set are started at
                // random locations"
                rndi = this.Rnd.Next(features.Count);
                for (i1 = rndi; i1 < features.Count; ++i1)
                {
                    if (TakeStep(i1, i2, features, classValues)) return 1;
                }
                for (i1 = 0; i1 < rndi; ++i1)
                {
                    if (TakeStep(i1, i2, features, classValues)) return 1;
                }
            } // if ((r2 < -tol && alph2 < C) || (r2 > tol && alph2 > 0))

            // "In extremely degenerate circumstances, none of the examples
            // will make an adequate second example. When this happens, the
            // first example is skipped and SMO continues with another chosen
            // first example."
            return 0;
        } 

        private double ComputeAll(List<double> vector,
          List<List<double>> features, List<int> classValues)
        {
            // output using all training data, even if alpha[] is zero
            double sum = -this.Bias;  // quirk of SMO paper
            for (int i = 0; i < features.Count; ++i)
            {
                if (this.Alpha[i] > 0)
                    sum += this.Alpha[i] * classValues[i] *
                      KernelFunc(features[i], vector);
            }
            return sum;
        }

    } // class SupportVectorMachine
}

