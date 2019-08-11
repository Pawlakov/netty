namespace ClickbaitGenerator.NeuralNet.Service.Layers
{
    using System.Collections.Generic;
    using System.Threading.Tasks;

    using ClickbaitGenerator.NeuralNet.Helpers;
    using ClickbaitGenerator.NeuralNet.Model;
    using ClickbaitGenerator.NeuralNet.Model.Contracts;
    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;
    using ClickbaitGenerator.NeuralNet.Service.Layers.Contracts;

    public class DeconvolutionLayer : IDecoderLayer
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

        public List<IMap> Maps { get; } = new List<IMap>();
        public int Height { get; private set; }
        public int Width { get; private set; }
        public int Size { get; private set; }
        public int FilterRadii { get; private set; }
        public int Step { get; private set; }
        public int Padding { get; private set; }

        public DeconvolutionLayer(ConvolutionLayer newConvolutionLayer)
        {
            this.ActivationActivationFunction = newConvolutionLayer.ActivationActivationFunction;
            this.Height = newConvolutionLayer.Height;
            this.Width = newConvolutionLayer.Width;
            this.Size = newConvolutionLayer.Size;
            this.FilterRadii = newConvolutionLayer.FilterRadii;
            this.Step = newConvolutionLayer.Step;
            this.Padding = newConvolutionLayer.Padding;

            this.ThisLayerID = ID;
            ID++;
        }

        public void ConnectNeurons(IDecoderLayer previousDecoderLayer)
        {
            this.Maps.Clear();
            this.FilterMaps.Clear();

            //Width = previousDecoderLayer.Width;
            //Height = previousDecoderLayer.Height;

            for (int i = 0; i < this.Size; i++)
            {
                this.FilterMaps.Add(new FilterMap(this.Width, this.Height, ConnectionHelper.AssignToConnectionBackwards, this.FilterRadii, this.Step, this.Padding));
                this.Maps.Add(this.FilterMaps[this.FilterMaps.Count - 1]);
                this.FilterMaps[this.FilterMaps.Count - 1].ConnectNeurons(previousDecoderLayer.Maps);
            }

            //Width = Maps[0].Width;
            //Height = Maps[0].Height;
        }

        public void ConnectNeurons(IOutputLayer previousLayer)
        {
            this.Maps.Clear();
            this.FilterMaps.Clear();

            this.Width = previousLayer.Width;
            this.Height = previousLayer.Height;
            for (int i = 0; i < this.Size; i++)
            {
                this.FilterMaps.Add(new FilterMap(this.Width, this.Height, ConnectionHelper.AssignToConnectionBackwards, this.FilterRadii, this.Step, this.Padding));
                this.Maps.Add(this.FilterMaps[this.FilterMaps.Count - 1]);
                this.FilterMaps[this.FilterMaps.Count - 1].ConnectNeurons(previousLayer.SourceImage);
            }

            this.Width = this.Maps[0].Width;
            this.Height = this.Maps[0].Height;
        }
        //TODO make parallel
        public void CalculateNeuronsActivation()
        {
            Parallel.ForEach(this.FilterMaps, (filterMap) => filterMap.CalculateNeuronsActivation(this.ActivationActivationFunction));
            //int mapsCount = Maps.Count;
            //for (int i = 0; i < mapsCount; i++)
            //{
            //    FilterMaps[i].CalculateNeuronsActivation(ActivationActivationFunction);
            //}
        }

        public override string ToString()
        {
            string msg = "Deconvolution Layer: " + this.ThisLayerID + "\n";

            foreach (var filterMap in this.FilterMaps)
            {
                msg += filterMap.ToString();
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
        //TODO make parallel - prototype
        public void Mutate(float learningFactor, float inertiaFactor)
        {
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