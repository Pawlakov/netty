namespace ClickbaitGenerator.NeuralNet.Model
{
    using System;
    using System.Collections.Generic;

    using ClickbaitGenerator.NeuralNet.Helpers;
    using ClickbaitGenerator.NeuralNet.Model.Contracts;
    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;

    public class FilterMap : IFilterMap
    {
        private static int ID = 0;
        public int ThisMapID{get; private set; }
        public float Bias { get; set; }
        public List<INeuron> Neurons { get; private set; } = new List<INeuron>();
        /// <summary>
        /// How will the connection be created - what neuron will be input, what will be the output.
        /// First neuron argument is the neuron of this map, second - neuron of map that is being connected to this one.
        /// </summary>
        private Action<INeuron, INeuron, Connection> _assignToConnection;

        public int Size => this.Neurons.Count;
        public int FilterRadii { get; }
        public int Step { get; }
        public int Padding { get; }
        public int Width { get; private set; }
        public int Height { get; private set; }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="height">Height of the source data (for example image).</param>
        /// <param name="AssignToConnection">How will the connection be created - what neuron will be input, what will be the output.
        /// First neuron argument is the neuron of this map, second - neuron of map that is being connected to this one.</param>
        /// <param name="filterRadii">Radius of the filter, not counting the middle.
        /// For 3x3 filter, set this to 1. For 5x5 set this to 2, etc.</param>
        /// <param name="step">How far away from each other the centers of the filters will be.
        /// Set to 1 if these should be one next to another.</param>
        /// <param name="padding">Width of the padding around the source layer. One unit = one neuron.</param>
        /// <param name="width">Width of the source info(for example image). One unit = one neuron.</param>
        public FilterMap(int width, int height, Action<INeuron, INeuron, Connection> AssignToConnection, int filterRadii = 1, int step = 1, int padding = 1)
        {
            this.FilterRadii = filterRadii;
            this.Step = step;
            this.Padding = padding;
            this.Width = width;
            this.Height = height;
            this._assignToConnection = AssignToConnection;
            this.Bias = CustomRandom.NextFloat();

            this.ThisMapID = ID;
            ID++;
        }

        /// <summary>
        /// Directly operates on passed neurons, connecting the layers.
        /// </summary>
        /// <param name="previousLayer">Neurons from the previous layer which we will connect to.</param>
        public void ConnectNeurons(List<IMap> previousLayer)
        {
            this.Neurons.Clear();

            Random random = new Random();
            var mapInputSize = previousLayer.Count;
           
            if (mapInputSize <= 0)
            {
                //throw new NeuralNetworkException("Passed map cannot be empty! Map size: ", mapInputSize); //Nothing to do here, welp...
            }
            

            int currentRow = -1;
            var thisNeuronsAmount = this.Width * this.Height;//All maps are the same anyways, in terms of size.
            
            this.ChkIfConnectionPossible(previousLayer[0].Neurons.Count);
            for (int i = 0; i < thisNeuronsAmount; i++)
            {
                var newNeuron = new Neuron((float)random.NextDouble());
                int iByWidth = i % this.Width; //What column are we currently at?
                if (iByWidth <= 0)
                {
                    currentRow++; //If the i is dividable by Width - we got to another row.
                }

                for (int column = iByWidth - this.Padding; column <= iByWidth + this.Padding; column++)
                {
                    for (int row = currentRow - this.Padding; row <= currentRow + this.Padding; row++)
                    {
                        if (column < 0 || row < 0 ||
                            row >= this.Height || //If the current row enters the bottom padding
                            column >= this.Width) //If the current column enters right padding
                        {
                            for (int filterMap = 0;
                                filterMap < previousLayer.Count;
                                filterMap++)
                            {
                                //newNeuron.AddConnection(new Connection(0.0f), assignSourceNeuronAs);//We are in the padding zone, add zeros.
                                this._assignToConnection(newNeuron, null, new Connection(0.0f));
                            } 
                        }
                        else
                        {
                            
                            int inputNeuronIndex = column + row * this.Width;
                            for (int filterMap = 0;
                                filterMap < previousLayer.Count;
                                filterMap++) //Since all source maps are the same - add connections to all maps on that position.
                            {
                                //connection.AssignInputNeuron(previousLayer[filterMap].Neurons[inputNeuronIndex]);
                                this._assignToConnection(newNeuron, previousLayer[filterMap].Neurons[inputNeuronIndex], new Connection(CustomRandom.NextFloat()));
                                //newNeuron.AddConnection(connection, assignSourceNeuronAs);
                            }
                        }
                    }
                }

                this.Neurons.Add(newNeuron);
            }
            

            //At the end, shrink the size properly.
            this.Width = this.Width + (this.Padding << 1) - (this.FilterRadii << 1);
            this.Width /= this.Step;
            this.Height = this.Height + (this.Padding << 1) - (this.FilterRadii << 1);
            this.Height /= this.Step;
        }

        public void ConnectNeurons(IMap previousLayer)
        {
            var tempList = new List<IMap>();
            tempList.Add(previousLayer);
            this.ConnectNeurons(tempList);
        }

        public void SetNeurons(List<INeuron> previousLayerNeurons)
        {
            this.Neurons = previousLayerNeurons;
        }

        public void CalculateNeuronsActivation(IActivationFunction activationActivationFunction)
        {
            for (int i = 0; i < this.Neurons.Count; i++)
            {
                var currentNeuron = this.Neurons[i];
                var thisNeuronActivation = activationActivationFunction.Calculate(currentNeuron.CalculateInputsSum() + this.Bias);
                currentNeuron.Activation = thisNeuronActivation;
            }
        }

        /// <summary>
        /// Defines whether is it possible to perfectly connect this layer
        /// with provided one. Perfectly == divide the neurons in the same
        /// way along whole source layer. Stores calculated value in _neuronAmount
        /// </summary>
        /// <param name="inputNeuronsAmount"></param>
        private void ChkIfConnectionPossible(int inputNeuronsAmount)
        {
            //The formula behind: (W-K+P)/S + 1 states that if the result is an integer,
            //then such neuron spacing can be applied. Otherwise, it cannot be applied.
            //W - amount of input neurons.
            //K - radius of the filter, not counting in the middle.
            //P - padding width.
            //S - Step between the filters. 1 means standing next to each other.
            float result = inputNeuronsAmount - this.FilterRadii + this.Padding;
            result = result / this.Step + 1;

            if (result != Math.Floor(result))
            {
                //It is not possible to setup the neurons in equal distances.
                //throw new NeuralNetworkException("Error - cannot equally position the neurons. Division result: ", result);
            }
        }

        public void ModifyNeuron(float value, int neuronID)
        {
            throw new NotImplementedException();
        }
        

        public override string ToString()
        {
            string msg = "Filter map: " + this.ThisMapID + "\n";

            foreach (var neuron in this.Neurons)
            {
                msg += "    " + neuron;
            }

            return msg;
        }
    }
}
