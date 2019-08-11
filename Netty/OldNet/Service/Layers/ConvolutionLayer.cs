namespace ClickbaitGenerator.NeuralNet.Service.Layers
{
    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;

    using ClickbaitGenerator.NeuralNet.Helpers;
    using ClickbaitGenerator.NeuralNet.Model;
    using ClickbaitGenerator.NeuralNet.Model.Contracts;
    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;
    using ClickbaitGenerator.NeuralNet.Service.Layers.Contracts;

    /// <summary>
    /// Layer that performs the convolution task over the input.
    /// </summary>
    public class ConvolutionLayer : IEncoderLayer
    {
        private static int ID = 0;
        /// <summary>
        /// The activation function used by this layer.
        /// </summary>
        public IActivationFunction ActivationActivationFunction { get; private set; }
        public int ThisLayerID{get;}
        /// <summary>
        /// True maps used by this layer.
        /// </summary>
        private List<IFilterMap> FilterMaps { get; set; } = new List<IFilterMap>();

        /// <summary>
        /// Maps that are visible outside. References to maps in FilterMaps, simply.
        /// </summary>
        public List<IMap> Maps { get; } = new List<IMap>();
        public int Size { get; private set; }
        public int FilterRadii { get; }
        public int Step { get; }
        public int Padding { get; }
        public int Width { get; private set; }
        public int Height { get; private set; }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="activationActivationFunction">Activation function that will be used by this layer.</param>
        /// <param name="filterRadii">Radius of the filter, not counting the middle.
        /// For 3x3 filter, set this to 1. For 5x5 set this to 2, etc.
        /// Padding becomes same as the filter (size of processed image stays the same).</param>
        /// <param name="filterMaps">Amount of filters used by this single layer.</param>
        public ConvolutionLayer(IActivationFunction activationActivationFunction, int filterRadii = 1, int filterMaps = 1)
        {
            this.ActivationActivationFunction = activationActivationFunction;
            this.FilterRadii = filterRadii;
            this.Step = 1;
            this.Padding = filterRadii;
            this.Size = filterMaps;

            this.ThisLayerID = ID;
            ID++;
        }

        public void ConnectNeurons(IInputLayer previousLayer)
        {
            this.Maps.Clear();
            this.FilterMaps.Clear();

            this.Width = previousLayer.Width;
            this.Height = previousLayer.Height;
            for (int i = 0; i < this.Size; i++)
            {
                this.FilterMaps.Add(new FilterMap(this.Width, this.Height, ConnectionHelper.AssignToConnection, this.FilterRadii, this.Step, this.Padding));
                this.Maps.Add(this.FilterMaps[this.FilterMaps.Count-1]);
                this.FilterMaps[this.FilterMaps.Count - 1].ConnectNeurons(previousLayer.SourceImage);
            }

            this.Width = this.Maps[0].Width;
            this.Height = this.Maps[0].Height;
        }
        //TODO made parallel - prototype
        public void CalculateNeuronsActivation()
        {
            Parallel.ForEach(this.FilterMaps, (filterMap) => filterMap.CalculateNeuronsActivation(this.ActivationActivationFunction));
            //int mapsCount = Maps.Count;
            //for (int i = 0; i < mapsCount; i++)
            //{
            //    FilterMaps[i].CalculateNeuronsActivation(ActivationActivationFunction);
            //}
        }

        /// <summary>
        /// Fills this layer with neurons , instantly connecting them to provided layer
        /// using provided through the constructor parameters.
        /// If parameters and provided layer size do not allow for equally positioned
        /// neurons - an exception is thrown (NeuralNetworkException).
        /// </summary>
        /// <param name="previousEncoderLayer">Layer which will this layer try connect to.</param>
        public void ConnectNeurons(IEncoderLayer previousEncoderLayer)
        {
            this.Maps.Clear();
            this.FilterMaps.Clear();

            this.Width = previousEncoderLayer.Width;
            this.Height = previousEncoderLayer.Height;

            for (int i = 0; i < this.Size; i++)
            {
                this.FilterMaps.Add(new FilterMap(this.Width, this.Height, ConnectionHelper.AssignToConnection, this.FilterRadii, this.Step, this.Padding));
                this.Maps.Add(this.FilterMaps[this.FilterMaps.Count - 1]);
                this.FilterMaps[this.FilterMaps.Count - 1].ConnectNeurons(previousEncoderLayer.Maps);
            }

            this.Width = this.Maps[0].Width;
            this.Height = this.Maps[0].Height;
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

        public override string ToString()
        {
            string msg = "Convolution Layer: " + this.ThisLayerID + "\n";

            foreach (var filterMap in this.FilterMaps)
            {
                msg += filterMap.ToString() + ',';
            }

            return msg;
        }
        //TODO make parallel - prototype
        public void CalculateDeltas()
        {
            var mapsAmount = this.FilterMaps.Count;
            Parallel.For(0, mapsAmount, i =>
            {
                var currentMap = this.FilterMaps[i];
                var mapNeurons = currentMap.Neurons;
                var neuronsInMap = mapNeurons.Count;
                for (int j = 0; j < neuronsInMap; j++)
                {
                    var currentNeuron = mapNeurons[j];
                    currentNeuron.Delta = currentNeuron.Activation * (1 - currentNeuron.Activation) *
                                          currentNeuron.CalculateOutputConnDeltasSum();
                }
            });
        }
        //TODO Make parallel - prototype
        public void Mutate(float learningFactor, float inertiaFactor)
        {
            //Mutate the output connections from all maps in this layer.
            var mapsCount = this.FilterMaps.Count;
            int neuronsAmount;
            Parallel.For(0, mapsCount, i =>
            {
                var neurons = this.FilterMaps[i].Neurons;
                neuronsAmount = neurons.Count;
                for (int j = 0; j < neuronsAmount; j++)
                {
                    var neuron = neurons[j];
                    neuron.MutateOutputConnections(learningFactor, inertiaFactor);
                }
            });


            //Mutate the biases for this layer.
            Parallel.For(0, mapsCount, i =>
            {
                var delta = 0.0f;
                var currentMap = this.FilterMaps[i];
                var neurons = currentMap.Neurons;
                neuronsAmount = neurons.Count;
                for (int j = 0; j < neuronsAmount; j++)
                {
                    delta += neurons[j].Delta;
                }

                currentMap.Bias += learningFactor * delta;
            });
        }
    }
}
