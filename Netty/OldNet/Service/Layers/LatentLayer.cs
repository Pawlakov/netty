namespace ClickbaitGenerator.NeuralNet.Service.Layers
{
    using System;
    using System.Collections.Generic;

    using ClickbaitGenerator.NeuralNet.Helpers;
    using ClickbaitGenerator.NeuralNet.Model;
    using ClickbaitGenerator.NeuralNet.Model.Contracts;
    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;
    using ClickbaitGenerator.NeuralNet.Service.Layers.Contracts;

    public class LatentLayer : ILatentLayer
    {
        /// <summary>
        /// Bias of the flattening (input) layer
        /// </summary>
        private float _flatteningBias;
        /// <summary>
        /// Bias of the latent (very middle, the smallest one, called Neurons) layer.
        /// </summary>
        private float _latentBias;
        /// <summary>
        /// Bias of the deflattening (output) layer.
        /// </summary>
        private float _deflatteningBias;
        private static int ID = 0;
        public int ThisLayerID{get;}
        /// <summary>
        /// The activation function used by this layer.
        /// </summary>
        public IActivationFunction ActivationActivationFunction { get; private set; }

        /// <summary>
        /// Stores the neurons that are used to flatten all maps from connected layer - input from encoder.
        /// </summary>
        public List<INeuron> FlattenInputNeurons { get; } = new List<INeuron>();

        /// <summary>
        /// Output to decoder.
        /// </summary>
        public List<INeuron> DeflattenLayer { get; } = new List<INeuron>();

        public int Height { get; }      //Height of the bottleneck 
        public int Width { get; } = 1; //This is always 1 column thick as we count the latent space only (bottleneck).
        public int LatentSpaceSize { get; }
        public List<INeuron> Neurons { get; } = new List<INeuron>();

        public LatentLayer(IActivationFunction activationActivationFunction, int latentSpaceSize)
        {
            this._flatteningBias = CustomRandom.NextFloat();
            this._latentBias = CustomRandom.NextFloat();
            this._deflatteningBias = CustomRandom.NextFloat();
            this.ActivationActivationFunction = activationActivationFunction;
            this.Height = latentSpaceSize;

            this.ThisLayerID = ID;
            ID++;
        }
        public void ModifyNeuron(float value, int neuronID)
        {
            throw new NotImplementedException();
        }

        public void ConnectNeurons(IEncoderLayer previousEncoderLayer)//Connect all neurons from source maps to all neurons of flattening layer.
        {
            this.FlattenInputNeurons.Clear();
            this.DeflattenLayer.Clear();
            this.Neurons.Clear();

            this.FormFlatteningLayer(previousEncoderLayer);
            this.FormBottleneckLayer();
            this.FormDeflatteningLayer();
        }

        public void ConnectOutputNeurons(IDecoderLayer outputEncoderLayer)
        {
            Random random = new Random();
            var outputMaps = outputEncoderLayer.Maps;
            int mapCount = outputMaps.Count;

            for (int i = 0; i < mapCount; i++)
            {
                int mapSize = outputMaps[i].Neurons.Count;
                var outputNeurons = outputMaps[i].Neurons;
                int deflattenSize = this.DeflattenLayer.Count;

                for (int j = 0; j < mapSize; j++)
                {
                    for (int k = 0; k < deflattenSize; k++)
                    {
                        var newConnection = new Connection(CustomRandom.NextFloat());
                        //newConnection.AssignInputNeuron(DeflattenLayer[k]);
                        ConnectionHelper.AssignToConnectionBackwards(this.DeflattenLayer[k], outputNeurons[j], newConnection);
                    }
                }
            }
        }
        //TODO make parallel
        public void CalculateNeuronsActivation()
        {
            int calculatedNeuronsCount = this.FlattenInputNeurons.Count;
            //Calculate activations for flattening layer
            for (int i = 0; i < calculatedNeuronsCount; i++)
            {
                this.FlattenInputNeurons[i].Activation = this.ActivationActivationFunction.Calculate(this.FlattenInputNeurons[i].CalculateInputsSum() + this._flatteningBias);
            }

            //calculate activations for middle layer
            calculatedNeuronsCount = this.Neurons.Count;
            for (int i = 0; i < calculatedNeuronsCount; i++)
            {
                this.Neurons[i].Activation = this.ActivationActivationFunction.Calculate(this.Neurons[i].CalculateInputsSum() + this._latentBias);
            }
            //Calculate activations for deflattening layer
            calculatedNeuronsCount = this.DeflattenLayer.Count;
            for (int i = 0; i < calculatedNeuronsCount; i++)
            {
                this.DeflattenLayer[i].Activation = this.ActivationActivationFunction.Calculate(this.DeflattenLayer[i].CalculateInputsSum() + this._deflatteningBias);
            }
        }

        /// <summary>
        /// Forms a flattening layer which neurons are connected to all source maps (all to all).
        /// </summary>
        /// <param name="previousEncoderLayer"></param>
        private void FormFlatteningLayer(IEncoderLayer previousEncoderLayer)
        {
            var sourceMaps = previousEncoderLayer.Maps;
            var sourceMapsSize = sourceMaps.Count;
            var random = new Random();

            for (int i = 0; i < sourceMapsSize; i++)
            {
                var mapNeurons = sourceMaps[i].Neurons;
                var mapNeuronsSize = mapNeurons.Count;
                for (int j = 0; j < mapNeuronsSize; j++)
                {
                    var newNeuron = new Neuron((float)random.NextDouble());

                    for (int currentMap = 0;
                        currentMap < sourceMapsSize;
                        currentMap++) //For all available source maps (which are of the same size)
                    {
                        for (int currentNeuron = 0;
                            currentNeuron < mapNeuronsSize;
                            currentNeuron++) //For all neurons in single map
                        {
                            var newConnection = new Connection(CustomRandom.NextFloat());
                            //newConnection.AssignInputNeuron(sourceMaps[currentMap].Neurons[currentNeuron]); //Add connections to all neurons on the currentNeuron positions on all source maps.
                            ConnectionHelper.AssignToConnection(newNeuron, sourceMaps[currentMap].Neurons[currentNeuron], newConnection);
                        }
                    }

                    this.FlattenInputNeurons.Add(newNeuron);
                }
            }
        }
        /// <summary>
        /// Using provided upon instantiation LatentSpaceSize, 
        /// </summary>
        private void FormBottleneckLayer()
        {
            var random = new Random();
            for (int i = 0; i < this.Height; i++)    //For each neuron in the bottleneck...
            {
                var newNeuron = new Neuron((float)random.NextDouble());
                for (int j = 0; j < this.FlattenInputNeurons.Count; j++)    //...add connections to ALL neurons from flattening layer.
                {
                    var connection = new Connection(CustomRandom.NextFloat());
                    //connection.AssignInputNeuron(FlattenInputNeurons[j]);
                    ConnectionHelper.AssignToConnection(newNeuron, this.FlattenInputNeurons[j], connection);
                }
                this.Neurons.Add(newNeuron);
            }

        }
        /// <summary>
        /// Sets the third part of the layer - deflattening layer. This will be the beginning of the decoder.
        /// </summary>
        private void FormDeflatteningLayer()
        {
            var random = new Random();
            for (int i = 0; i < this.FlattenInputNeurons.Count; i++)    //For each new neuron in deflattening layer (which has the same size as flattening one)...
            {
                var newNeuron = new Neuron((float)random.NextDouble());
                for (int j = 0; j < this.Neurons.Count; j++) //...Add connections to ALL neurons in bottleneck sublayer.
                {
                    var connection = new Connection(CustomRandom.NextFloat());
                    //connection.AssignInputNeuron(Neurons[j]);
                    ConnectionHelper.AssignToConnection(newNeuron, this.Neurons[j], connection);
                }
                this.DeflattenLayer.Add(newNeuron);
            }

        }

        public override string ToString()
        {
            string msg = "Latent Layer: " + this.ThisLayerID + "\n";

            msg += "    Flattening sublayer: \n";
            foreach (var neuron in this.FlattenInputNeurons)
            {
                msg += neuron.ToString();
            }
            msg += "    Latent sublayer: \n";
            foreach (var neuron in this.Neurons)
            {
                msg += neuron.ToString();
            }
            msg += "    Deflattening sublayer: \n";
            foreach (var neuron in this.DeflattenLayer)
            {
                msg += neuron.ToString();
            }

            return msg;
        }

        public List<float> GetLatentValues()
        {
            var neuronsInLayer = this.Neurons.Count;
            var neuronIndex = 0;
            var output = new List<float>(neuronsInLayer);

            for (int i = 0; i < neuronsInLayer; i++)//For each input value
            {
                //Read activation values of all neurons.
                output.Add(this.Neurons[neuronIndex].Activation);
            }
            return output;
        }

        public void SetLatentLayer(List<float> input)
        {
            if (input.Count != Neurons.Count)
            {
                //throw new NeuralNetworkException("Passed amount of values differs from neurons amount! These have to be of same quantity!" +
                //                                 $"\n Passed values count: {input.Count}, Neurons count: {Neurons.Count}.");
            }

            for (int i = 0; i < input.Count; i++)
            {
                Neurons[i].Activation = input[i];
            }
        }

        //TODO make parallel
        public void Mutate(float learningFactor, float inertiaFactor)
        {
            var deflattenLayerCount = this.DeflattenLayer.Count;
            INeuron currentNeuron;
            var biasDelta = 0.0f;
            for (int i = 0; i < deflattenLayerCount; i++)
            {
                currentNeuron = this.DeflattenLayer[i];
                currentNeuron.MutateOutputConnections(learningFactor, inertiaFactor);
                biasDelta += currentNeuron.Delta;
            }

            this._deflatteningBias += learningFactor * biasDelta;

            var latentLayerCount = this.Neurons.Count;
            biasDelta = 0.0f;
            for (int i = 0; i < latentLayerCount; i++)
            {
                currentNeuron = this.Neurons[i];
                currentNeuron.MutateOutputConnections(learningFactor, inertiaFactor);
                biasDelta += currentNeuron.Delta;
            }

            this._latentBias += learningFactor * biasDelta;

            var flatteningLayerCount = this.FlattenInputNeurons.Count;
            biasDelta = 0.0f;
            for (int i = 0; i < flatteningLayerCount; i++)
            {
                currentNeuron = this.FlattenInputNeurons[i];
                currentNeuron.MutateOutputConnections(learningFactor, inertiaFactor);
                biasDelta += currentNeuron.Delta;
            }

            this._flatteningBias += learningFactor * biasDelta;
        }

        public void CalculateDeltas()
        {
            var neuronsCount = this.DeflattenLayer.Count;
            for (int i = 0; i < neuronsCount; i++)
            {
                var currentNeuron = this.DeflattenLayer[i];
                currentNeuron.Delta = currentNeuron.Activation * (1 - currentNeuron.Activation) *currentNeuron.CalculateOutputConnDeltasSum();
            }

            neuronsCount = this.Neurons.Count;
            for (int i = 0; i < neuronsCount; i++)
            {
                var currentNeuron = this.Neurons[i];
                currentNeuron.Delta = currentNeuron.Activation * (1 - currentNeuron.Activation) * currentNeuron.CalculateOutputConnDeltasSum();
            }

            neuronsCount = this.FlattenInputNeurons.Count;
            for (int i = 0; i < neuronsCount; i++)
            {
                var currentNeuron = this.FlattenInputNeurons[i];
                currentNeuron.Delta = currentNeuron.Activation * (1 - currentNeuron.Activation) * currentNeuron.CalculateOutputConnDeltasSum();
            }
        }
    }
}
