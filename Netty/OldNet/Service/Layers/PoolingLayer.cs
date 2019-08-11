namespace ClickbaitGenerator.NeuralNet.Service.Layers
{
    using System.Collections.Generic;
    using System.Threading.Tasks;

    using ClickbaitGenerator.NeuralNet.Model;
    using ClickbaitGenerator.NeuralNet.Model.Contracts;
    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;
    using ClickbaitGenerator.NeuralNet.Service.Layers.Contracts;

    public class PoolingLayer : IEncoderLayer
    {
        private static int ID = 0;
        public int ThisLayerID{get;}
        /// <summary>
        /// The activation function used by this layer.
        /// </summary>
        public IPoolingFunction ActivationFunction { get; private set; }

        /// <summary>
        /// Stores max indexes found during pooling of connected maps from previous layer.
        /// Outer list is for maps, inner lists group indexes of max values in these maps.
        /// </summary>
        public List<List<int>> MaxIndexesInMaps { get; private set; } = new List<List<int>>();
        /// <summary>
        /// The divisor that will be used to divide the Height and Width of
        /// source layer.
        /// </summary>
        public int Divisor { get; }
        public int Height { get; private set; }
        public int Width { get; private set; }
        public int InputWidth { get; private set; }
        public int InputHeight { get; private set; }

        public int Size
        {
            get { return this.PoolingMaps.Count; }
        }

        public List<IMap> Maps { get; } = new List<IMap>();

        private List<IPoolingMap> PoolingMaps { get; } = new List<IPoolingMap>();
        //public List<INeuron> Neurons { get; } = new List<INeuron>();

        /// <summary>
        /// 
        /// </summary>
        /// <param name="activationFunction">Activation function that will be used for this layer.</param>
        /// <param name="divisor">The divisor that will be used to divide the Height and Width of the source layer.</param>
        public PoolingLayer(IPoolingFunction activationFunction, int divisor = 2)
        {
            this.ActivationFunction = activationFunction;
            this.Divisor = divisor;

            this.ThisLayerID = ID;
            ID++;
        }
        public void ConnectNeurons(IEncoderLayer previousEncoderLayer)
        {
            this.Width = previousEncoderLayer.Width;    //Setup he base size - of previous layer. Will be used to generate maps.
            this.Height = previousEncoderLayer.Height;
            this.InputWidth = this.Width;
            this.InputHeight = this.Height;
            this.PoolingMaps.Clear();

            for (int i = 0; i < previousEncoderLayer.Maps.Count; i++)
            {
                var newPoolingMap = new PoolingMap(this.Width, this.Height, this.Divisor);
                newPoolingMap.ConnectNeurons(previousEncoderLayer.Maps[i]);
                this.PoolingMaps.Add(newPoolingMap);
                this.Maps.Add(newPoolingMap);
            }

            this.Width = this.Maps[0].Width;      //Assign new width and height of maps of this layer.
            this.Height = this.Maps[0].Height;

            this.CreateIndexesLists();       //At the end, create proper max indexes lists.
        }

        public void ConnectNeurons(IInputLayer previousLayer)
        {
            this.Width = previousLayer.Width;    //Setup he base size - of previous layer. Will be used to generate maps.
            this.Height = previousLayer.Height;
            this.InputWidth = this.Width;
            this.InputHeight = this.Height;
            this.PoolingMaps.Clear();
            
            var newPoolingMap = new PoolingMap(this.Width, this.Height, this.Divisor);
            newPoolingMap.ConnectNeurons(previousLayer.SourceImage);
            this.PoolingMaps.Add(newPoolingMap);
            this.Maps.Add(newPoolingMap);

            this.Width = this.Maps[0].Width;      //Assign new width and height of maps of this layer.
            this.Height = this.Maps[0].Height;

            this.CreateIndexesLists();       //At the end, create proper max indexes lists.
        }
        //TODO make parallel
        public void CalculateNeuronsActivation()
        {
            int mapsCount = this.Maps.Count;
            Parallel.For(0, mapsCount, i =>
            {
                this.PoolingMaps[i].CalculateNeuronsActivation(this.ActivationFunction, this.MaxIndexesInMaps[i]);
            });
            //for (int i = 0; i < mapsCount; i++)
            //{
            //    PoolingMaps[i].CalculateNeuronsActivation(ActivationActivationFunction, MaxIndexesInMaps[i]);
            //}
        }

        public override string ToString()
        {
            string msg = "Pooling Layer: " + this.ThisLayerID + "\n";

            foreach (var map in this.PoolingMaps)
            {
                msg += map.ToString();
            }

            return msg;
        }

        public void CalculateDeltas()
        {
            var poolingMapsCount = this.PoolingMaps.Count;
            for (int i = 0; i < poolingMapsCount; i++)
            {
                var poolingNeurons = this.PoolingMaps[i].Neurons;
                var poolingMapNeuronsCount = poolingNeurons.Count;
                for (int j = 0; j < poolingMapNeuronsCount; j++)
                {
                    var currentNeuron = poolingNeurons[j];
                    currentNeuron.Delta = currentNeuron.Activation*(1-currentNeuron.Activation)*currentNeuron.CalculateOutputConnDeltasSum();
                }
            }
        }

        public void Mutate(float learningFactor, float inertiaFactor)
        {
            var mapsCount = this.PoolingMaps.Count;
            for (int i = 0; i < mapsCount; i++)
            {
                var neurons = this.PoolingMaps[i].Neurons;
                var neuronsAmount = neurons.Count;
                for (int j = 0; j < neuronsAmount; j++)
                {
                    var neuron = neurons[j];
                    neuron.MutateOutputConnections(learningFactor, inertiaFactor);
                }
            }
        }

        private void CreateIndexesLists()
        {
            var singleMapNeuronsAmount = this.PoolingMaps[0].Neurons.Count;
            for (int i = 0; i < this.PoolingMaps.Count; i++)
            {
                this.MaxIndexesInMaps.Add(new List<int>());
                for (int j = 0; j < singleMapNeuronsAmount; j++)
                {
                    this.MaxIndexesInMaps[i].Add(0);             //Fill the max index tables for each map with existing instances.
                                                            //Their amount won't change at all for the network so this will make
                                                            //it faster than constantly clearing the list and adding new instances.
                }
            }
        }
    }
}
