namespace ClickbaitGenerator.NeuralNet.Service.Layers
{
    using System;
    using System.Collections.Generic;

    using ClickbaitGenerator.NeuralNet.Helpers;
    using ClickbaitGenerator.NeuralNet.Model;
    using ClickbaitGenerator.NeuralNet.Model.Contracts;
    using ClickbaitGenerator.NeuralNet.Service.Layers.Contracts;

    public class InputLayer : IInputLayer
    {
        public int ThisLayerID { get; }
        private static int ID = 0;

        public int Height { get; private set; }
        public int Width { get; private set; }
        public List<IMap> SourceImage { get; private set; } = new List<IMap>();

        public InputLayer()
        {
            this.ThisLayerID = ID;
            ID++;
        }
        /// <summary>
        /// Allocates new maps containing neurons. Make sure the amount == width*height.
        /// </summary>
        /// <param name="amount">Amount of neurons in single map.</param>
        /// <param name="width">Width of a single map.</param>
        /// <param name="height">Height of a single map.</param>
        /// <param name="mapsAmount">Quantity of maps.</param>
        public void SetNeurons(int amount, int width, int height, int mapsAmount)
        {
            if (mapsAmount < 0)
            {
                //throw new NeuralNetworkException("There cannot be less than 1 map in a layer!");
            }
            if (width * height != amount)
            {
                //throw new NeuralNetworkException("Provided width and height do not equal the amount of neurons!", amount);
            }

            Random random = new Random();
            this.Width = width;
            this.Height = height;

            for (int j = 0; j < mapsAmount; j++)
            {
                var neurons = new List<INeuron>();
                for (int i = 0; i < amount; i++)
                {
                    var newNeuron = new Neuron((float) random.NextDouble());
                    neurons.Add(newNeuron);
                }

                IFilterMap map = new FilterMap(this.Width, this.Height, ConnectionHelper.AssignToConnection);
                map.SetNeurons(neurons);
                this.SourceImage.Add(map);
            }
        }
        /// <summary>
        /// Sets input for neurons. For input of multiple channels values should be provided for
        /// each pixel like so: RGB RGB RGB ... where each letter is a given color channel, and one float
        /// value of range from 0 to 1.
        /// </summary>
        /// <param name="data"></param>
        public void SetInput(List<float> data)
        {
            var dataCount = data.Count;
            var mapsCount = this.SourceImage.Count;
            var neuronsInLayer = mapsCount * this.SourceImage[0].Neurons.Count;

            if (neuronsInLayer != dataCount)
            {
                //throw new NeuralNetworkException($"The provided input data is of different length than declared for the input layer! \n Input: {dataCount} Declared: {neuronsInLayer}");
            }

            var neuronIndex = 0;
            for (int i = 0; i < dataCount; i+= mapsCount)//For each input value
            {
                for (int j = 0; j < mapsCount; j++)     //For each map
                {
                    //Set neurons on same position of different maps to given pixel partial value.
                    this.SourceImage[j].Neurons[neuronIndex].Activation = data[i + j]; 
                }

                neuronIndex++;
                //mapNeurons[i].Activation = data[i];
            }
        }


        public override string ToString()
        {
            string msg = "Input layer: \n";
            foreach (var map in this.SourceImage)
            {
                msg += map.ToString();
            }
            
            return msg;
        }
        //TODO make parallel
        public void Mutate(float learningFactor, float inertiaFactor)
        {
            for (int i = 0; i < this.SourceImage.Count; i++)
            {
                var neurons = this.SourceImage[i].Neurons;
                var neuronsAmount = neurons.Count;
                for (int j = 0; j < neuronsAmount; j++)
                {
                    var neuron = neurons[j];
                    neuron.MutateOutputConnections(learningFactor, inertiaFactor);
                }
            }
        }
        
    }
}
