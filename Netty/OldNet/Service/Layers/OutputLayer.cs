namespace ClickbaitGenerator.NeuralNet.Service.Layers
{
    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;

    using ClickbaitGenerator.NeuralNet.Model;
    using ClickbaitGenerator.NeuralNet.Model.Contracts;
    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;
    using ClickbaitGenerator.NeuralNet.Service.Layers.Contracts;

    public class OutputLayer : IOutputLayer, IMutatingOutputLayer
    {
        public int ThisLayerID { get; }
        private static int _id;
        
        public IActivationFunction ActivationActivationFunction { get;}
        public int Height { get; }
        public int Width { get; }
        public int Size { get; }
        public List<IMap> SourceImage { get; } = new List<IMap>();
        private List<OutputMap> OutputImage { get; } = new List<OutputMap>();


        public OutputLayer(IActivationFunction activationActivationFunction, int width, int height, int layersAmount)
        {
            this.ActivationActivationFunction = activationActivationFunction;
            Random random = new Random();
            this.Size = width * height;
            this.Width = width;
            this.Height = height;

            for (int i = 0; i < layersAmount; i++)
            {
                var neurons = new List<INeuron>();
                for (int j = 0; j < this.Size; j++)
                {
                    var newNeuron = new Neuron((float) random.NextDouble());
                    neurons.Add(newNeuron);
                }
                var output = new OutputMap(width, height);
                output.SetNeurons(neurons);
                this.SourceImage.Add(output);
                this.OutputImage.Add(output);
            }


            this.ThisLayerID = _id;
            _id++;
        }
        public void ConnectNeurons(List<IMap> neurons)
        {
            foreach (var map in this.SourceImage)
            {
                map.ConnectNeurons(neurons);
            }
        }
        //TODO prototype of parallelism
        public void CalculateNeuronsActivation()
        {
            Parallel.ForEach(this.OutputImage, (map) => map.CalculateNeuronsActivation(this.ActivationActivationFunction));
            //foreach (var map in OutputImage)
            //{
            //    map.CalculateNeuronsActivation(ActivationActivationFunction);
            //}
        }

        public List<float> GetOutput()
        {
            var mapsCount = this.SourceImage.Count;
            var neuronsInLayer = mapsCount * this.SourceImage[0].Neurons.Count;
            var neuronIndex = 0;
            var output = new List<float>(neuronsInLayer);

            for (int i = 0; i < neuronsInLayer; i += mapsCount)//For each input value
            {
                for (int j = 0; j < mapsCount; j++)     //For each map
                {
                    //Each map contains part of a same pixel. Read them.
                    output.Add(this.SourceImage[j].Neurons[neuronIndex].Activation);
                }

                neuronIndex++;
            }
            return output;
        }

        public override string ToString()
        {
            string msg = "Output layer: \n";
            msg += this.SourceImage.ToString() ;
            

            return msg;
        }

        public void CalculateDeltas(List<float> errorGradient)
        {
            var mapsCount = this.SourceImage.Count;
            //There should be same amount of errors an neurons.
            var neuronsInLayer = errorGradient.Count;//mapsCount * SourceImage[0].Neurons.Count;
            var neuronIndex = 0;

            for (int i = 0; i < neuronsInLayer; i += mapsCount)//For each input value
            {
                for (int j = 0; j < mapsCount; j++)     //For each map
                {
                    var map = this.SourceImage[j];
                    var currentNeuron = map.Neurons[neuronIndex];
                    currentNeuron.Delta = errorGradient[i+j] * currentNeuron.Activation * (1 - currentNeuron.Activation);
                }

                neuronIndex++;
            }



            
            //foreach (var map in SourceImage)
            //{
            //    var neurons = map.Neurons;
            //    for (int j = 0; j < errorsAmount; j++)
            //    {
            //        var currentNeuron = neurons[j];
            //        currentNeuron.Delta = errorGradient[j] * currentNeuron.Activation * (1 - currentNeuron.Activation);
            //    }
            //}
        }
    }
}
