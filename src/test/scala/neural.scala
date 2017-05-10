import org.scalatest.FunSuite

class NeuralTesting extends FunSuite {

  import com.rho.neural.types.{Signals}
  import com.rho.neural.basics.{Neuron, Layer}
  import com.rho.neural.networks.{PerceptronNetwork, SigmoidNetwork}

  test("XOR Gate, Perceptron Network") {
    val layer1 = Array( (Array(0.6,0.6),-1.0), (Array(1.1,1.1),-1.0) )
    val layer2 = Array( (Array(-2.0,1.1),-1.0) )
    val layers = Array(Layer(layer1),Layer(layer2))
    val network = new PerceptronNetwork(layers)
    assert(0===network.predict(Signals(0.0,0.0)).head)
    assert(1===network.predict(Signals(1.0,0.0)).head)
    assert(1===network.predict(Signals(0.0,1.0)).head)
    assert(0===network.predict(Signals(1.0,1.0)).head)
  }

  test("XOR Gate, Sigmoid Network") {
    val layer1 = Array( (Array(1.0,1.0),0.0), (Array(2.0,2.0),0.0) )
    val layer2 = Array( (Array(-1000.0,850.0),0.0) )
    val layers = Array(Layer(layer1),Layer(layer2))
    val network = new SigmoidNetwork(layers)
    assert(0.25>network.predict(Signals(0.0,0.0)).head)
    assert(0.75<network.predict(Signals(1.0,0.0)).head)
    assert(0.75<network.predict(Signals(0.0,1.0)).head)
    assert(0.25>network.predict(Signals(1.0,1.0)).head)
  }

  test("Deltas") {
    val layer1 = Array( (Signals(1.0,-7.0),0.0), (Signals(6.0,-4.0),0.0) )
    val layer2 = Array( (Signals(-1.0,23.0),0.0) ) 
    val layers = Array(Layer(layer1),Layer(layer2))
    val network = new SigmoidNetwork(layers) 
    val trainData = List(
      (Signals(0.0,0.0), Signals(0.0)),
      (Signals(1.0,0.0), Signals(1.0)),
      (Signals(0.0,1.0), Signals(1.0)),
      (Signals(1.0,1.0), Signals(0.0))
    )
  }

}
