using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ML_2017ACNTT
{
    public partial class MainForm : Form
    {
        #region Init
        public string trainFile, testFile;
        float[][] trainData, validData, testData;
        double[][] trainA, trainAT, trainATA, trainILamda, trainY, matrixWeight;  //train
        double[][] validA, validY, testA, testY;  //valid
        double step, lamdaMin, lamdaMax, bestLamda;
        List<double> lamdaVal, valLossVal, trainLossVal;
        int nTest, nVal, nTrain, nFeature, nMinIndex;
        double minVal, maxVal;

        public MainForm()
        {
            InitializeComponent();
            lamdaMin = float.Parse(tbLamdaMin.Text);
            lamdaMax = double.Parse(tbLamdaMax.Text);
            step = float.Parse(tbLamdaStep.Text);
            nTrain = 67; nTest = 5; nVal = 5; nFeature = 8;
            trainData = new float[nTrain][];
            validData = new float[nVal][];
            testData = new float[nTest][];
            trainA = new double[nTrain][];
            validA = new double[nVal][];
            testA = new double[nTest][];

            lamdaVal = new List<double>();
            valLossVal = new List<double>();
            trainLossVal = new List<double>();

            trainFile = tbTrain.Text;
            testFile = tbTest.Text;
            LoadTrainTestData();
            CalculateMatrixTrain();
            LoadMatrixValidTest();
        }
        #endregion

        #region Reload
        private void buttonReload_Click(object sender, EventArgs e)
        {
            trainFile = tbTrain.Text;
            testFile = tbTest.Text;
            LoadTrainTestData();
            CalculateMatrixTrain();
            LoadMatrixValidTest();
        }
        private void LoadTrainTestData()
        {
            //Read train data
            int nLine = 0;
            using (var reader = new StreamReader(trainFile))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');
                    if (nLine > 0)
                    {
                        trainData[nLine - 1] = Array.ConvertAll(values, s => float.Parse(s));
                    }
                    nLine++;
                }
            }

            //Read valid and test data
            nLine = 0;
            using (var reader = new StreamReader(testFile))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');
                    if (nLine <= 4)
                    {
                        validData[nLine] = Array.ConvertAll(values, s => float.Parse(s));
                    }
                    else
                        testData[nLine - 5] = Array.ConvertAll(values, s => float.Parse(s));

                    nLine++;
                }
            }
        }

        private void CalculateMatrixTrain()
        {
            trainY = MatrixCal.MatrixCreate(nTrain, 1);

            for (int i = 0; i < nTrain; i++)
            {
                trainA[i] = new double[nFeature + 1];
                trainA[i][0] = 1;
                for (int j = 0; j < 8; j++)
                {
                    trainA[i][j + 1] = trainData[i][j];
                }
                trainY[i][0] = trainData[i][8];
            }

            trainAT = MatrixCal.MatrixTranspose(trainA);
            trainATA = MatrixCal.MatrixProduct(trainAT, trainA);
        }

        private void LoadMatrixValidTest()
        {
            validY = MatrixCal.MatrixCreate(nVal, 1);

            for (int i = 0; i < nVal; i++)
            {
                validA[i] = new double[nFeature + 1];
                validA[i][0] = 1;
                for (int j = 0; j < 8; j++)
                {
                    validA[i][j + 1] = validData[i][j];
                }
                validY[i][0] = validData[i][8];
            }

            testY = MatrixCal.MatrixCreate(nTest, 1);

            for (int i = 0; i < nTest; i++)
            {
                testA[i] = new double[nFeature + 1];
                testA[i][0] = 1;
                for (int j = 0; j < 8; j++)
                {
                    testA[i][j + 1] = testData[i][j];
                }
                testY[i][0] = testData[i][8];
            }
        }

        #endregion

        #region Draw
        private void buttonDraw_Click(object sender, EventArgs e)
        {
            lamdaVal.Clear();
            valLossVal.Clear();
            trainLossVal.Clear();
            double lamda;
            for (lamda = lamdaMin; lamda <= lamdaMax; lamda += step)
            {
                CalculateRidgeRegression(lamda);
                CalculateLoss(lamda);
            }
            DrawChart();
        }
        private void CalculateRidgeRegression(double ld)
        {
            trainILamda = MatrixCal.MatrixIdentity(nFeature + 1, ld,false);
            double[][] matrixSum = MatrixCal.MatrixSum(trainATA, trainILamda);
            double[][] matrixSumInv = MatrixCal.MatrixInverse(matrixSum);
            double[][] matrixProd = MatrixCal.MatrixProduct(matrixSumInv, trainAT);
            matrixWeight = MatrixCal.MatrixProduct(matrixProd, trainY);
        }

        private void CalculateLoss(double ld)
        {
            lamdaVal.Add(ld);
            //train loss
            double trainLoss = 0;
            for (int i = 0; i < nTrain; i++)
            {
                double[][] trainAi = new double[1][];
                trainAi[0] = new double[nFeature + 1];
                for (int j = 0; j < nFeature + 1; j++)
                {
                    trainAi[0][j] = trainA[i][j];
                }

                double[][] trainAiw = MatrixCal.MatrixProduct(trainAi, matrixWeight);
                trainLoss += Math.Pow(trainY[i][0] - trainAiw[0][0], 2);
            }
            trainLossVal.Add(trainLoss/(double)nTrain);

            //valid loss
            double validLoss = 0;
            for (int i = 0; i < nVal; i++)
            {
                double[][] valAi = new double[1][];
                valAi[0] = new double[nFeature + 1];
                for (int j = 0; j < nFeature + 1; j++)
                {
                    valAi[0][j] = validA[i][j];
                }

                double[][] valAiw = MatrixCal.MatrixProduct(valAi, matrixWeight);
                validLoss += Math.Pow(validY[i][0] - valAiw[0][0], 2);
            }

            valLossVal.Add(validLoss / (double)nVal);

        }

        private void DrawChart()
        {
            chartData.Series[0].Points.Clear();
            chartData.Series[1].Points.Clear();
            minVal = valLossVal.Min();
            maxVal = valLossVal.Max();
            nMinIndex = valLossVal.IndexOf(valLossVal.Min());
            bestLamda = lamdaVal[nMinIndex];
            labelMinLamda.Text = "best lamda: " + bestLamda.ToString();

            double range = maxVal - minVal;

            chartData.ChartAreas[0].AxisY.Minimum = minVal - range / (double)10;
            chartData.ChartAreas[0].AxisY.Maximum = maxVal + range / (double)10;
            for (int n = 0; n < lamdaVal.Count; n++)
            {
                //chartData.Series[0].Points.AddXY(lamdaVal[n], trainLossVal[n]);
                chartData.Series[1].Points.AddXY(lamdaVal[n], valLossVal[n]);
            }
        }

        #endregion

        #region Calculate Result
        private void buttonCal_Click(object sender, EventArgs e)
        {
            if (lamdaVal.Count == 0 || trainLossVal.Count == 0 || valLossVal.Count == 0)
                MessageBox.Show("Draw first!");
            else
                CalculateResult();
        }
        private void CalculateResult()
        {
            CalculateRidgeRegression(lamdaVal[nMinIndex]);
            testY = MatrixCal.MatrixCreate(nTest, 1);
            labelResult.Text = "";

            for (int i = 0; i < nTest; i++)
            {
                double[][] testAi = new double[1][];
                testAi[0] = new double[nFeature + 1];
                for (int j = 0; j < nFeature + 1; j++)
                {
                    testAi[0][j] = testA[i][j];
                }

                double[][] testAiw = MatrixCal.MatrixProduct(testAi, matrixWeight);
                testY[i][0] = testAiw[0][0];
                labelResult.Text += testY[i][0].ToString() + "\n";
            }
        }

        #endregion

        #region Control
        private void tbLamdaMax_TextChanged(object sender, EventArgs e)
        {
            lamdaMax = double.Parse(tbLamdaMax.Text);
        }
        private void tbLamda_TextChanged(object sender, EventArgs e)
        {
            lamdaMin = double.Parse(tbLamdaMin.Text);
        }

        private void tbLamdaStep_TextChanged(object sender, EventArgs e)
        {
            step = double.Parse(tbLamdaStep.Text);
        }
        #endregion

    }
}
