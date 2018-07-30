using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML_2017ACNTT
{
    public class MatrixCal
    {
        public static double[] Solve(double[][] design)
        {
            // find linear regression coefficients
            // 1. peel off X matrix and Y vector
            int rows = design.Length;
            int cols = design[0].Length;
            double[][] X = MatrixCreate(rows, cols - 1);
            double[][] Y = MatrixCreate(rows, 1); // a column vector

            int j;
            for (int i = 0; i < rows; ++i)
            {
                for (j = 0; j < cols - 1; ++j)
                {
                    X[i][j] = design[i][j];
                }
                Y[i][0] = design[i][j]; // last column
            }

            // 2. B = inv(Xt * X) * Xt * y
            double[][] Xt = MatrixTranspose(X);
            double[][] XtX = MatrixProduct(Xt, X);
            double[][] inv = MatrixInverse(XtX);
            double[][] invXt = MatrixProduct(inv, Xt);

            double[][] mResult = MatrixProduct(invXt, Y);
            double[] result = MatrixToVector(mResult);
            return result;
        } // Solve

        static double RSquared(double[][] data, double[] coef)
        {
            // 'coefficient of determination'
            int rows = data.Length;
            int cols = data[0].Length;

            // 1. compute mean of y
            double ySum = 0.0;
            for (int i = 0; i < rows; ++i)
                ySum += data[i][cols - 1]; // last column
            double yMean = ySum / rows;

            // 2. sum of squared residuals & tot sum squares
            double ssr = 0.0;
            double sst = 0.0;
            double y; // actual y value
            double predictedY; // using the coef[] 
            for (int i = 0; i < rows; ++i)
            {
                y = data[i][cols - 1]; // get actual y

                predictedY = coef[0]; // start w/ intercept constant
                for (int j = 0; j < cols - 1; ++j) // j is col of data
                    predictedY += coef[j + 1] * data[i][j]; // careful

                ssr += (y - predictedY) * (y - predictedY);
                sst += (y - yMean) * (y - yMean);
            }

            if (sst == 0.0)
                throw new Exception("All y values equal");
            else
                return 1.0 - (ssr / sst);
        }

        public static double[][] Design(double[][] data)
        {
            // add a leading col of 1.0 values
            int rows = data.Length;
            int cols = data[0].Length;
            double[][] result = MatrixCreate(rows, cols + 1);
            for (int i = 0; i < rows; ++i)
                result[i][0] = 1.0;

            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result[i][j + 1] = data[i][j];

            return result;
        }

        // ===== Matrix routines

        public static double[][] MatrixCreate(int rows, int cols)
        {
            // allocates/creates a matrix initialized to all 0.0
            // do error checking here
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];
            return result;
        }


        static double[] MatrixToVector(double[][] matrix)
        {
            // single column matrix to vector
            int rows = matrix.Length;
            int cols = matrix[0].Length;
            if (cols != 1)
                throw new Exception("Bad matrix");
            double[] result = new double[rows];
            for (int i = 0; i < rows; ++i)
                result[i] = matrix[i][0];
            return result;
        }


        public static double[][] MatrixIdentity(int n, double value = 1.0, bool remove00 = true)
        {
            // return an n x n Identity matrix
            double[][] result = MatrixCreate(n, n);
            if (remove00)
                result[0][0] = 0;
            else
                result[0][0] = value;
            for (int i = 1; i < n; ++i)
                result[i][i] = value;

            return result;
        }

        // -------------------------------------------------------------

        public static double[][] MatrixProduct(double[][] matrixA, double[][] matrixB)
        {
            int aRows = matrixA.Length; int aCols = matrixA[0].Length;
            int bRows = matrixB.Length; int bCols = matrixB[0].Length;
            if (aCols != bRows)
                return null;
            //throw new Exception("Non-conformable matrices in MatrixProduct");

            double[][] result = MatrixCreate(aRows, bCols);

            for (int i = 0; i < aRows; ++i) // each row of A
                for (int j = 0; j < bCols; ++j) // each col of B
                    for (int k = 0; k < aCols; ++k) // could use k < bRows
                        result[i][j] += matrixA[i][k] * matrixB[k][j];

            //Parallel.For(0, aRows, i =>
            //  {
            //    for (int j = 0; j < bCols; ++j) // each col of B
            //      for (int k = 0; k < aCols; ++k) // could use k < bRows
            //        result[i][j] += matrixA[i][k] * matrixB[k][j];
            //  }
            //);

            return result;
        }

        // -------------------------------------------------------------

        public static double[][] MatrixSum(double[][] matrixA, double[][] matrixB)
        {
            int aRows = matrixA.Length; int aCols = matrixA[0].Length;
            int bRows = matrixB.Length; int bCols = matrixB[0].Length;
            if ((aCols != bCols) || (aRows != bRows))
                return null;

            double[][] result = MatrixCreate(aRows, aCols);

            for (int i = 0; i < aRows; ++i) // each row of A
                for (int j = 0; j < aCols; ++j) // each col of B
                    result[i][j] = matrixA[i][j] + matrixB[i][j];

            return result;
        }

        // -------------------------------------------------------------

        static double[] MatrixVectorProduct(double[][] matrix, double[] vector)
        {
            // result of multiplying an n x m matrix by a m x 1 column vector (yielding an n x 1 column vector)
            int mRows = matrix.Length; int mCols = matrix[0].Length;
            int vRows = vector.Length;
            if (mCols != vRows)
                throw new Exception("Non-conformable matrix and vector in MatrixVectorProduct");
            double[] result = new double[mRows]; // an n x m matrix times a m x 1 column vector is a n x 1 column vector
            for (int i = 0; i < mRows; ++i)
                for (int j = 0; j < mCols; ++j)
                    result[i] += matrix[i][j] * vector[j];
            return result;
        }


        static double[][] MatrixDecompose(double[][] matrix, out int[] perm,
          out int toggle)
        {
            // Doolittle LUP decomposition with partial pivoting.
            // returns: result is L (with 1s on diagonal) and U;
            // perm holds row permutations; toggle is +1 or -1 (even or odd)
            int rows = matrix.Length;
            int cols = matrix[0].Length;

            if (rows != cols)
            {
                toggle = 0;
                perm = null;
                return null;
            }
            //throw new Exception("Non-square mattrix");

            int n = rows; // convenience

            double[][] result = MatrixDuplicate(matrix); // 

            perm = new int[n]; // set up row permutation result
            for (int i = 0; i < n; ++i) { perm[i] = i; }

            toggle = 1; // toggle tracks row swaps

            for (int j = 0; j < n - 1; ++j) // each column
            {
                double colMax = Math.Abs(result[j][j]);
                int pRow = j;
                //for (int i = j + 1; i < n; ++i) // deprecated
                //{
                //  if (result[i][j] > colMax)
                //  {
                //    colMax = result[i][j];
                //    pRow = i;
                //  }
                //}

                for (int i = j + 1; i < n; ++i) // reader Matt V needed this:
                {
                    if (Math.Abs(result[i][j]) > colMax)
                    {
                        colMax = Math.Abs(result[i][j]);
                        pRow = i;
                    }
                }
                // Not sure if this approach is needed always, or not.

                if (pRow != j) // if largest value not on pivot, swap rows
                {
                    double[] rowPtr = result[pRow];
                    result[pRow] = result[j];
                    result[j] = rowPtr;

                    int tmp = perm[pRow]; // and swap perm info
                    perm[pRow] = perm[j];
                    perm[j] = tmp;

                    toggle = -toggle; // adjust the row-swap toggle
                }

                // -------------------------------------------------------------
                // This part added later (not in original code) 
                // and replaces the 'return null' below.
                // if there is a 0 on the diagonal, find a good row 
                // from i = j+1 down that doesn't have
                // a 0 in column j, and swap that good row with row j

                if (result[j][j] == 0.0)
                {
                    // find a good row to swap
                    int goodRow = -1;
                    for (int row = j + 1; row < n; ++row)
                    {
                        if (result[row][j] != 0.0)
                            goodRow = row;
                    }

                    if (goodRow == -1)
                        return null;
                    // throw new Exception("Cannot use Doolittle's method");

                    // swap rows so 0.0 no longer on diagonal
                    double[] rowPtr = result[goodRow];
                    result[goodRow] = result[j];
                    result[j] = rowPtr;

                    int tmp = perm[goodRow]; // and swap perm info
                    perm[goodRow] = perm[j];
                    perm[j] = tmp;

                    toggle = -toggle; // adjust the row-swap toggle
                }
                // -------------------------------------------------------------

                //if (Math.Abs(result[j][j]) < 1.0E-20) // deprecated
                //  return null; // consider a throw

                for (int i = j + 1; i < n; ++i)
                {
                    result[i][j] /= result[j][j];
                    for (int k = j + 1; k < n; ++k)
                    {
                        result[i][k] -= result[i][j] * result[j][k];
                    }
                }

            } // main j column loop

            return result;
        } // MatrixDecompose

        // -------------------------------------------------------------

        public static double[][] MatrixInverse(double[][] matrix)
        {
            int n = matrix.Length;
            double[][] result = MatrixDuplicate(matrix);

            int[] perm;
            int toggle;
            double[][] lum = MatrixDecompose(matrix, out perm, out toggle);
            if (lum == null)
                return null;
            // throw new Exception("Unable to compute inverse");

            double[] b = new double[n];
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (i == perm[j])
                        b[j] = 1.0;
                    else
                        b[j] = 0.0;
                }

                double[] x = HelperSolve(lum, b); // use decomposition

                for (int j = 0; j < n; ++j)
                    result[j][i] = x[j];
            }
            return result;
        }

        // -------------------------------------------------------------

        public static double[][] MatrixTranspose(double[][] matrix)
        {
            int rows = matrix.Length;
            int cols = matrix[0].Length;
            double[][] result = MatrixCreate(cols, rows); // note indexing
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    result[j][i] = matrix[i][j];
                }
            }
            return result;
        } // TransposeMatrix


        static double MatrixDeterminant(double[][] matrix)
        {
            int[] perm;
            int toggle;
            double[][] lum = MatrixDecompose(matrix, out perm, out toggle);
            if (lum == null)
                throw new Exception("Unable to compute MatrixDeterminant");
            double result = toggle;
            for (int i = 0; i < lum.Length; ++i)
                result *= lum[i][i];
            return result;
        }

        // -------------------------------------------------------------

        static double[] HelperSolve(double[][] luMatrix, double[] b)
        {
            // before calling this helper, permute b using the perm array
            // from MatrixDecompose that generated luMatrix
            int n = luMatrix.Length;
            double[] x = new double[n];
            b.CopyTo(x, 0);

            for (int i = 1; i < n; ++i)
            {
                double sum = x[i];
                for (int j = 0; j < i; ++j)
                    sum -= luMatrix[i][j] * x[j];
                x[i] = sum;
            }

            x[n - 1] /= luMatrix[n - 1][n - 1];
            for (int i = n - 2; i >= 0; --i)
            {
                double sum = x[i];
                for (int j = i + 1; j < n; ++j)
                    sum -= luMatrix[i][j] * x[j];
                x[i] = sum / luMatrix[i][i];
            }

            return x;
        }


        static double[][] MatrixDuplicate(double[][] matrix)
        {
            // allocates/creates a duplicate of a matrix
            double[][] result = MatrixCreate(matrix.Length, matrix[0].Length);
            for (int i = 0; i < matrix.Length; ++i) // copy the values
                for (int j = 0; j < matrix[i].Length; ++j)
                    result[i][j] = matrix[i][j];
            return result;
        }
    }
}
