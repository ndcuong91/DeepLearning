namespace ML_2017ACNTT
{
    partial class MainForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea1 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend1 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series1 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.Series series2 = new System.Windows.Forms.DataVisualization.Charting.Series();
            this.chartData = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.tbTrain = new System.Windows.Forms.TextBox();
            this.tbTest = new System.Windows.Forms.TextBox();
            this.tbLamdaMin = new System.Windows.Forms.TextBox();
            this.buttonDraw = new System.Windows.Forms.Button();
            this.tbLamdaStep = new System.Windows.Forms.TextBox();
            this.buttonCal = new System.Windows.Forms.Button();
            this.labelStep = new System.Windows.Forms.Label();
            this.labelLamda = new System.Windows.Forms.Label();
            this.labelResult = new System.Windows.Forms.Label();
            this.buttonReload = new System.Windows.Forms.Button();
            this.tbLamdaMax = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.labelMinLamda = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.chartData)).BeginInit();
            this.SuspendLayout();
            // 
            // chartData
            // 
            chartArea1.Name = "ChartArea1";
            this.chartData.ChartAreas.Add(chartArea1);
            legend1.Name = "Legend1";
            this.chartData.Legends.Add(legend1);
            this.chartData.Location = new System.Drawing.Point(12, 119);
            this.chartData.Name = "chartData";
            this.chartData.Palette = System.Windows.Forms.DataVisualization.Charting.ChartColorPalette.Excel;
            series1.ChartArea = "ChartArea1";
            series1.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Point;
            series1.Legend = "Legend1";
            series1.Name = "TrainingLoss";
            series2.ChartArea = "ChartArea1";
            series2.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Point;
            series2.Legend = "Legend1";
            series2.Name = "ValidLoss";
            this.chartData.Series.Add(series1);
            this.chartData.Series.Add(series2);
            this.chartData.Size = new System.Drawing.Size(1119, 498);
            this.chartData.TabIndex = 0;
            this.chartData.Text = "chart1";
            this.chartData.UseWaitCursor = true;
            // 
            // tbTrain
            // 
            this.tbTrain.Location = new System.Drawing.Point(26, 11);
            this.tbTrain.Name = "tbTrain";
            this.tbTrain.Size = new System.Drawing.Size(330, 20);
            this.tbTrain.TabIndex = 26;
            this.tbTrain.Text = "D:\\2.Cao hoc\\Ky 2\\Hoc may\\1-training-data.csv";
            // 
            // tbTest
            // 
            this.tbTest.Location = new System.Drawing.Point(26, 37);
            this.tbTest.Name = "tbTest";
            this.tbTest.Size = new System.Drawing.Size(330, 20);
            this.tbTest.TabIndex = 28;
            this.tbTest.Text = "D:\\2.Cao hoc\\Ky 2\\Hoc may\\Nguyen-Duy-Cuong-test.csv";
            // 
            // tbLamdaMin
            // 
            this.tbLamdaMin.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tbLamdaMin.Location = new System.Drawing.Point(478, 8);
            this.tbLamdaMin.Name = "tbLamdaMin";
            this.tbLamdaMin.Size = new System.Drawing.Size(60, 27);
            this.tbLamdaMin.TabIndex = 29;
            this.tbLamdaMin.Text = "0";
            this.tbLamdaMin.TextChanged += new System.EventHandler(this.tbLamda_TextChanged);
            // 
            // buttonDraw
            // 
            this.buttonDraw.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(192)))), ((int)(((byte)(192)))));
            this.buttonDraw.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.buttonDraw.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonDraw.Location = new System.Drawing.Point(561, 8);
            this.buttonDraw.Name = "buttonDraw";
            this.buttonDraw.Size = new System.Drawing.Size(77, 44);
            this.buttonDraw.TabIndex = 30;
            this.buttonDraw.Text = "Draw";
            this.buttonDraw.UseVisualStyleBackColor = false;
            this.buttonDraw.Click += new System.EventHandler(this.buttonDraw_Click);
            // 
            // tbLamdaStep
            // 
            this.tbLamdaStep.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tbLamdaStep.Location = new System.Drawing.Point(478, 83);
            this.tbLamdaStep.Name = "tbLamdaStep";
            this.tbLamdaStep.Size = new System.Drawing.Size(60, 27);
            this.tbLamdaStep.TabIndex = 32;
            this.tbLamdaStep.Text = "0.01";
            this.tbLamdaStep.TextChanged += new System.EventHandler(this.tbLamdaStep_TextChanged);
            // 
            // buttonCal
            // 
            this.buttonCal.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(255)))), ((int)(((byte)(192)))));
            this.buttonCal.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.buttonCal.Font = new System.Drawing.Font("Microsoft Sans Serif", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonCal.Location = new System.Drawing.Point(561, 66);
            this.buttonCal.Name = "buttonCal";
            this.buttonCal.Size = new System.Drawing.Size(77, 44);
            this.buttonCal.TabIndex = 33;
            this.buttonCal.Text = "Calculate";
            this.buttonCal.UseVisualStyleBackColor = false;
            this.buttonCal.Click += new System.EventHandler(this.buttonCal_Click);
            // 
            // labelStep
            // 
            this.labelStep.AutoSize = true;
            this.labelStep.Location = new System.Drawing.Point(425, 90);
            this.labelStep.Name = "labelStep";
            this.labelStep.Size = new System.Drawing.Size(29, 13);
            this.labelStep.TabIndex = 34;
            this.labelStep.Text = "Step";
            // 
            // labelLamda
            // 
            this.labelLamda.AutoSize = true;
            this.labelLamda.Location = new System.Drawing.Point(409, 15);
            this.labelLamda.Name = "labelLamda";
            this.labelLamda.Size = new System.Drawing.Size(58, 13);
            this.labelLamda.TabIndex = 35;
            this.labelLamda.Text = "Lamda min";
            // 
            // labelResult
            // 
            this.labelResult.AutoSize = true;
            this.labelResult.Font = new System.Drawing.Font("Microsoft Sans Serif", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelResult.Location = new System.Drawing.Point(669, 30);
            this.labelResult.Name = "labelResult";
            this.labelResult.Size = new System.Drawing.Size(45, 15);
            this.labelResult.TabIndex = 36;
            this.labelResult.Text = "Result:";
            // 
            // buttonReload
            // 
            this.buttonReload.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(192)))), ((int)(((byte)(192)))));
            this.buttonReload.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.buttonReload.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonReload.Location = new System.Drawing.Point(26, 66);
            this.buttonReload.Name = "buttonReload";
            this.buttonReload.Size = new System.Drawing.Size(330, 37);
            this.buttonReload.TabIndex = 37;
            this.buttonReload.Text = "Reload";
            this.buttonReload.UseVisualStyleBackColor = false;
            this.buttonReload.Click += new System.EventHandler(this.buttonReload_Click);
            // 
            // tbLamdaMax
            // 
            this.tbLamdaMax.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tbLamdaMax.Location = new System.Drawing.Point(478, 44);
            this.tbLamdaMax.Name = "tbLamdaMax";
            this.tbLamdaMax.Size = new System.Drawing.Size(60, 27);
            this.tbLamdaMax.TabIndex = 38;
            this.tbLamdaMax.Text = "5";
            this.tbLamdaMax.TextChanged += new System.EventHandler(this.tbLamdaMax_TextChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(408, 51);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(61, 13);
            this.label1.TabIndex = 39;
            this.label1.Text = "Lamda max";
            // 
            // labelMinLamda
            // 
            this.labelMinLamda.AutoSize = true;
            this.labelMinLamda.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelMinLamda.Location = new System.Drawing.Point(668, 9);
            this.labelMinLamda.Name = "labelMinLamda";
            this.labelMinLamda.Size = new System.Drawing.Size(76, 16);
            this.labelMinLamda.TabIndex = 40;
            this.labelMinLamda.Text = "Best lamda";
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(192)))), ((int)(((byte)(255)))), ((int)(((byte)(192)))));
            this.ClientSize = new System.Drawing.Size(1143, 629);
            this.Controls.Add(this.labelMinLamda);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.tbLamdaMax);
            this.Controls.Add(this.buttonReload);
            this.Controls.Add(this.labelResult);
            this.Controls.Add(this.labelLamda);
            this.Controls.Add(this.labelStep);
            this.Controls.Add(this.buttonCal);
            this.Controls.Add(this.tbLamdaStep);
            this.Controls.Add(this.buttonDraw);
            this.Controls.Add(this.tbLamdaMin);
            this.Controls.Add(this.tbTest);
            this.Controls.Add(this.tbTrain);
            this.Controls.Add(this.chartData);
            this.Name = "MainForm";
            this.Text = "ML 2017ACNTT";
            ((System.ComponentModel.ISupportInitialize)(this.chartData)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.DataVisualization.Charting.Chart chartData;
        private System.Windows.Forms.TextBox tbTrain;
        private System.Windows.Forms.TextBox tbTest;
        private System.Windows.Forms.TextBox tbLamdaMin;
        private System.Windows.Forms.Button buttonDraw;
        private System.Windows.Forms.TextBox tbLamdaStep;
        private System.Windows.Forms.Button buttonCal;
        private System.Windows.Forms.Label labelStep;
        private System.Windows.Forms.Label labelLamda;
        private System.Windows.Forms.Label labelResult;
        private System.Windows.Forms.Button buttonReload;
        private System.Windows.Forms.TextBox tbLamdaMax;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label labelMinLamda;
    }
}

