namespace ClickbaitGenerator.NeuralNet.Model
{
    using System;
    using System.Collections.Generic;

    using ClickbaitGenerator.NeuralNet.Helpers;
    using ClickbaitGenerator.NeuralNet.Model.Contracts;
    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;

    public class PoolingMap : IPoolingMap
    {
        private static int ID = 0;
        public int ThisMapID{get; private set;}

        public float Bias { get; set; }
        public int Height { get; private set; }
        public int Width { get; private set; }
        public List<INeuron> Neurons { get; private set; } = new List<INeuron>();
        /// <summary>
        /// The divisor that will be used to divide the Height and Width of
        /// source layer.
        /// </summary>
        private int _divisor { get; }

        public PoolingMap(int width, int height, int divisor = 2)
        {
            this.Width = width;
            this.Height = height;
            this._divisor = divisor;
            this.Bias = CustomRandom.NextFloat();

            this.ThisMapID = ID;
            ID++;
        }
        public void ModifyNeuron(float value, int neuronID)
        {
            throw new NotImplementedException();
        }
        

        public void ConnectNeurons(IMap previousLayer)
        {
            Random random = new Random();
            
            int currentRow = 0;
            int columnCounter = 0;//Counts how many columns has the algorythm already passed. If too many - resets itself and goes to another row.


            var singleMapNeuronsCount = previousLayer.Neurons.Count;    //Amount of neurons in each map (all maps are the same)

            for (int i = 0; i < singleMapNeuronsCount; i += this._divisor, columnCounter += this._divisor)
            {
                var newNeuron = new Neuron((float)random.NextDouble());
                int iByWidth; //What column are we currently at?
                if (columnCounter >= this.Width)
                {
                    iByWidth = 0; //We begin new row - position yourself at first column
                    if (columnCounter > this.Width) //If column counter is bigger than Width - Width is odd, so we have to subtract 1 from i
                    {
                        //in order to keep the filter windows aligned.
                        i--;
                    }

                    i += (this.Width * (this._divisor - 1)); //jump Divisor-1 rows

                    if (i >= singleMapNeuronsCount)
                    {
                        break; //If the main counter (source neurons) gets out of bounds - stop. We have covered all of
                    } //source neurons.

                    columnCounter = 0;
                    currentRow += this._divisor; //If the i is dividable by Width - we got to another row.
                }
                else
                {
                    iByWidth = i % this.Width;
                }

                for (int column = iByWidth; column < iByWidth + this._divisor; column++)
                {
                    for (int row = currentRow; row < currentRow + this._divisor; row++)
                    {
                        if (column >= this.Width || row >= this.Height) //If given filter is only partially covered
                        {
                            //newNeuron.AddConnection(new Connection(0.0f), connectSourceAs); //We are in the padding zone, add zeros.
                            ConnectionHelper.AssignToConnection(newNeuron, null, new Connection(0.0f));
                        }
                        else
                        {
                            var connection = new Connection(CustomRandom.NextFloat());
                            int inputNeuronIndex = column + row * this.Width;
                            //connection.AssignInputNeuron(previousLayer.Neurons[inputNeuronIndex]);
                            ConnectionHelper.AssignToConnection(newNeuron, previousLayer.Neurons[inputNeuronIndex], connection);
                            //newNeuron.AddConnection(connection, connectSourceAs);
                        }
                    }
                }

                this.Neurons.Add(newNeuron);
            }


            //At the end, shrink the size properly.
            this.Width = (int)Math.Ceiling((float)previousLayer.Width / this._divisor); //Width and height are either equal to or bigger than the result (up to closest integer number)
            this.Height = (int)Math.Ceiling((float)previousLayer.Height / this._divisor);
        }

        public void CalculateNeuronsActivation(IPoolingFunction activationFunction, List<int> maxIndexes)
        {
            var neuronsAmount = this._divisor * this._divisor;
            var currentWindow = new List<float>(neuronsAmount);//Size of a single window is _divisor*_divisor

            for (int i = 0; i < neuronsAmount; i++)
            {
                currentWindow.Add(0.0f);                //Populate the list with objects.
            }

            int neuronsCount = this.Neurons.Count;

            maxIndexes.Clear();

            for (int i = 0; i < neuronsCount; i++)
            {
                var inputConnections = this.Neurons[i].InputConnections;
                var inputConnectionsCount = inputConnections.Count;
                for (int j = 0; j < inputConnectionsCount; j++)
                {
                    var inputNeuron = inputConnections[j].InputNeuron;
                    if(inputNeuron != null)
                    { 
                        currentWindow[j] = inputNeuron.Activation; //inputConnections[j].Weight.Value;
                    }
                    else
                    {
                        currentWindow[j] = -1.0f;                  //There's no neuron connected - do not even bother with this connection.
                    }
                    inputConnections[j].Weight.Value = 0.0f;                    //Reset all input connections weights.
                }

                var maxIndex = 0;
                this.Neurons[i].Activation = activationFunction.Calculate(currentWindow, out maxIndex);
                inputConnections[maxIndex].Weight.Value = 1.0f;                 //Make connection with max activation have non-changing weight. Also America great again.
                maxIndexes.Add(maxIndex);       //Remember what neuron had the highest activation value.
            }

        }

        //Not nice one... But well.
        public void ConnectNeurons(List<IMap> previousLayer)
        {
            if (previousLayer.Count > 0)
            {
                this.ConnectNeurons(previousLayer[0]);
            }
        }

        public void SetNeurons(List<INeuron> previousLayerNeurons)
        {
            this.Neurons = previousLayerNeurons;
        }

        public override string ToString()
        {
            string msg = "Pooling map: " + this.ThisMapID + "\n";

            foreach (var neuron in this.Neurons)
            {
                msg += "    " + neuron;
            }

            return msg;
        }
    }
}
