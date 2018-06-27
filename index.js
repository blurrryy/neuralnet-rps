function quickSort(arr) {
  if (arr.length <= 1) return arr;
  let piv = arr[arr.length - 1];
  let l = [];
  let r = [];
  for (let i = 0; i < arr.length - 1; i++) {
    arr[i].fitness < piv.fitness ? l.push(arr[i]) : r.push(arr[i]);
  }
  return [...quickSort(l), piv, ...quickSort(r)];
}

class NNCounter {
  static add() {
    if (!this.idx) {
      this.idx = 0;
    }
    this.idx++;
    return this.idx;
  }
}

class NeuralNetwork {
  constructor(i, h, o) {
    let inputLayer = this.createLayer(i);
    let hiddenLayer = this.createLayer(h);
    let outputLayer = this.createLayer(o);
    let inputWeights = this.createWeights(i, hiddenLayer);
    let hiddenWeights = this.createWeights(h, outputLayer);
    let nn = [
      inputLayer,
      inputWeights,
      hiddenLayer,
      hiddenWeights,
      outputLayer
    ];
    this.neuralNetwork = nn;
    this.initWeights();
    this.fitness = 0;
    this.alive = 0;
    this.wins = 0;
    this.draws = 0;
    this.mutated = 0;
    this.id = NNCounter.add();
    this.winrate = 0;
    this.cycle = 0;
  }

  recalculateWinrate() {
    let wR = 0;
    wR = ~~((this.wins / (this.alive + 1)) * 100) / 10;
    this.winrate = wR;
  }

  createLayer(len) {
    let layer = [];
    for (let i = 0; i < len; i++) {
      layer.push(0);
    }
    return layer;
  }

  createWeights(len, layer) {
    let weightLayer = [];
    for (let i = 0; i < len; i++) {
      weightLayer.push(layer);
    }
    return weightLayer;
  }

  initWeights() {
    let i = [1, 3];
    let neuralNetwork = this.neuralNetwork;
    let newNN = [];
    newNN.push(neuralNetwork[0]);
    for (let e of i) {
      let iWeights = [];
      let nn = neuralNetwork[e];
      for (let iNeuron = 0; iNeuron < nn.length; iNeuron++) {
        let iNeu = [];
        for (
          let singleWeight = 0;
          singleWeight < nn[iNeuron].length;
          singleWeight++
        ) {
          iNeu.push(this.getRandomWeight());
        }
        iWeights.push(iNeu);
      }
      newNN.push(iWeights);
      if (e == 1) newNN.push(neuralNetwork[2]);
    }
    newNN.push(neuralNetwork[4]);
    this.neuralNetwork = [];
    this.neuralNetwork = newNN;
  }

  getRandomWeight() {
    return Math.random() * 10 - 5;
  }

  alterInputs(inp) {
    let nn = this.neuralNetwork;
    let newNN = [];
    newNN.push(inp);
    for (let i = 1; i < nn.length; i++) {
      newNN.push(nn[i]);
    }
    this.neuralNetwork = [];
    this.neuralNetwork = newNN;
  }

  calculateLayer(start, bias) {
    let input = this.neuralNetwork[start];
    let weigths = this.neuralNetwork[start + 1];
    let newHiddenLayer = [];
    for (let t = 0; t < this.neuralNetwork[start + 2].length; t++) {
      let xcalc = [];
      for (let f = 0; f < input.length; f++) {
        xcalc.push(input[f] * weigths[f][t]);
      }
      xcalc.push(-bias);
      let hLayNeuron = 0;
      for (let el of xcalc) {
        hLayNeuron = hLayNeuron + el;
      }
      hLayNeuron = this.sigmoid(hLayNeuron);
      newHiddenLayer.push(hLayNeuron);
    }
    return newHiddenLayer;
  }

  cross(xNN) {
    let w1 = this.neuralNetwork[1];
    let w2 = this.neuralNetwork[3];
    for (let w = 0; w < w1.length; w++) {
      for (let u = 0; u < w1[w].length; u++) {
        if (Math.random() > 0.995) {
          w1[w][u] = this.getRandomWeight();
          continue;
        }
        if (Math.random() > 0.5) {
          w1[w][u] = xNN.neuralNetwork[1][w][u];
        }
      }
    }
    for (let w = 0; w < w2.length; w++) {
      for (let u = 0; u < w2[w].length; u++) {
        if (Math.random() > 0.995) {
          w2[w][u] = this.getRandomWeight();
          continue;
        }
        if (Math.random() > 0.5) {
          w2[w][u] = xNN.neuralNetwork[3][w][u];
        }
      }
    }
    this.mutated = 1;
    return this;
  }

  sigmoid(t) {
    return 1 / (1 + Math.pow(Math.E, -t));
  }

  calculate() {
    let nn = this.neuralNetwork;
    this.neuralNetwork[2] = this.calculateLayer(0, 0);
    this.neuralNetwork[4] = this.calculateLayer(2, 0);
    return this.neuralNetwork[4];
  }

  next() {
    let o = this.calculate();
    this.lastChoice = o.indexOf(Math.max(...o)) + 1;
    return this.lastChoice;
  }

  checkWin(playerChoice) {
    this.alive++;
    this.cycle++;
    if (this.cycle > this.neuralNetwork[0].length ** 2) this.fitness /= 2;
    if (this.lastChoice == playerChoice) {
      this.fitness -= 10;
      this.draws++;
      return;
    }
    if (playerChoice == 1 && this.lastChoice == 3) {
      this.fitness += 10;
      this.wins++;
      return;
    }
    if (playerChoice == 3 && this.lastChoice == 2) {
      this.fitness += 10;
      this.wins++;
      return;
    }
    if (playerChoice == 2 && this.lastChoice == 1) {
      this.fitness += 10;
      this.wins++;
      return;
    }
    this.fitness -= 15;
  }
}

class NetworkX {
  constructor(i, hist) {
    this.networkX = [];
    for (let c = 0; c < i; c++) {
      this.networkX.push(new NeuralNetwork(hist, 9, 3));
    }
    this.history = [];
    for (let c = 0; c < hist; c++) {
      this.history.push(0);
    }
    for (let nn of this.networkX) {
      nn.alterInputs(this.history);
      nn.next();
    }
  }

  strongestNet() {
    return [
      this.networkX[0].lastChoice,
      this.networkX[0].fitness,
      this.networkX[0].alive + 1,
      this.networkX[0].mutated,
      this.networkX[0].id,
      this.networkX[0].winrate
    ];
  }

  inputUserChoice(i) {
    for (let cnn of this.networkX) {
      cnn.checkWin(i);
    }
    this.networkX = [...quickSort(this.networkX)].reverse();
    for (let cnn of this.networkX) {
      console.log(
        "Win Percentage: " +
          Math.floor((cnn.wins / cnn.alive) * 100) +
          "% || Fitness: " +
          cnn.fitness
      );
    }
    this.evolution();
    this.updateInputs(i);
  }

  evolution() {
    let cnt = this.networkX.length;
    let kp = ~~(cnt * 0.7);
    let newNN = this.networkX.slice(0);
    newNN.splice(kp, newNN.length - kp);
    this.ev(newNN);
    let freshNN = [];
    for (let c = 0; c < this.networkX.length - kp; c++) {
      freshNN.push(
        new NeuralNetwork(this.networkX[0].neuralNetwork[0].length, 9, 3)
      );
    }
    this.networkX = [...newNN, ...freshNN];
  }

  ev(nn) {
    for (let c = 2; c < nn.length; c++) {
      if (Math.random() > 0.8) {
        nn[c].cross(nn[~~(Math.random() * ~~(nn.length / 2))]);
      }
    }
  }

  updateInputs(i) {
    this.history.push(i);
    this.history.shift();
    for (let nn of this.networkX) {
      nn.recalculateWinrate();
      nn.alterInputs(this.history);
      nn.next();
    }
  }
}
