namespace ClickbaitGenerator.NeuralNet.Service.Layers
{
    using System.Collections.Generic;
    using System.Threading.Tasks;

    using ClickbaitGenerator.NeuralNet.Model;
    using ClickbaitGenerator.NeuralNet.Model.Contracts;
    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;
    using ClickbaitGenerator.NeuralNet.Service.Layers.Contracts;

    public class UnpoolingLayer : IDecoderLayer
    {
        private static int ID = 0;
        public int ThisLayerID{get;}
        
        public int Height { get; }
        public int Width { get; }
        public int Size { get; }
        private int _divisor { get; set; }
        public List<IMap> Maps { get; } = new List<IMap>();
        private List<IUnpoolingMap> UnpoolingMaps { get; } = new List<IUnpoolingMap>();
        private readonly List<List<int>> _maxIndexesInMaps;
        /// <summary>
        /// The activation function used by this layer.
        /// </summary>
        public IActivationFunction ActivationActivationFunction { get; private set; }


        public UnpoolingLayer(IActivationFunction activationActivationFunction, PoolingLayer correspondingLayer)
            :this(activationActivationFunction,
                correspondingLayer.MaxIndexesInMaps,
                correspondingLayer.Width, 
                correspondingLayer.Height, 
                correspondingLayer.Size, 
                correspondingLayer.Divisor)
        {
            
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="activationActivationFunction">Used to calculate this layers neurons' activations.</param>
        /// <param name="maxIndexesInMaps">Synchronizes this layer with proper pooling layer.
        /// Indexes of max values found in pooled maps.</param>
        /// <param name="width">Width of input of corresponding pooling map.</param>
        /// <param name="height">height of input of corresponding pooling map.</param>
        /// <param name="size">Amount of maps in one layer.</param>
        /// <param name="divisor">Defines scale factor - by this value width and height will be multiplied
        ///  and by this value squared the previous' layer's maps will be enlarged.</param>
        public UnpoolingLayer(IActivationFunction activationActivationFunction, List<List<int>> maxIndexesInMaps, int width, int height, int size, int divisor)
        {
            this.ActivationActivationFunction = activationActivationFunction;
            this._maxIndexesInMaps = maxIndexesInMaps;
            this.Width = width;
            this.Height = height;
            this._divisor = divisor;
            this.Size = size;

            for (int i = 0; i < this.Size; i++)
            {
                this.UnpoolingMaps.Add(new UnpoolingMap(this.Width, this.Height, this._divisor));
                this.Maps.Add(this.UnpoolingMaps[this.UnpoolingMaps.Count -1]);
            }

            this.ThisLayerID = ID;
            ID++;
        }

        public void ConnectNeurons(IDecoderLayer previousEncoderLayer)
        {
            var inputMaps = previousEncoderLayer.Maps;
            for (int i = 0; i < inputMaps.Count; i++)
            {
                //var newUnpoolingMap = new FilterMap(Height, Width);
                this.Maps[i].ConnectNeurons(inputMaps[i]);
                //Maps.Add(newUnpoolingMap);
            }
        }

        public void ConnectNeurons(IOutputLayer previousLayer)
        {
            //var newUnpoolingMap = new FilterMap(Height, Width);
            this.Maps[0].ConnectNeurons(previousLayer.SourceImage);
            //Maps.Add(newUnpoolingMap);
        }
        //TODO make parallel
        public void CalculateNeuronsActivation()
        {
            int mapsCount = this.UnpoolingMaps.Count;
            Parallel.For(0, mapsCount, i =>
            {
                this.UnpoolingMaps[i].CalculateNeuronsActivation(this._maxIndexesInMaps[i], this.ActivationActivationFunction);
            });
            //    for (int i = 0; i < UnpoolingMaps.Count; i++)
            //{
            //    UnpoolingMaps[i].CalculateNeuronsActivation(_maxIndexesInMaps[i], ActivationActivationFunction);
            //}
        }

        public override string ToString()
        {
            string msg = "Unpooling Layer: " + this.ThisLayerID + "\n";

            foreach (var map in this.Maps)
            {
                msg += map.ToString();
            }
            

            return msg;
        }

        public void CalculateDeltas()
        {
            var mapsAmount = this.UnpoolingMaps.Count;
            for (int i = 0; i < mapsAmount; i++)
            {
                var mapNeurons = this.UnpoolingMaps[i].Neurons;
                var neuronsInMap = mapNeurons.Count;
                for (int j = 0; j < neuronsInMap; j++)
                {
                    var currentNeuron = mapNeurons[j];
                    currentNeuron.Delta = currentNeuron.Activation * (1 - currentNeuron.Activation) *
                                          currentNeuron.CalculateOutputConnDeltasSum();
                }
            }
        }

        public void Mutate(float learningFactor, float inertiaFactor)
        {
            //No need for mutation. I know, I'm a bad programmer...
        }
    }
}
