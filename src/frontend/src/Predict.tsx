import { useState } from 'react';

function App() {

  interface InputTypes{
    m: number;
    mt: number;
    ec: number;
    ep: number;
    fuelC: number;
  }

  const [inputData, setInputData] = useState<InputTypes>({ m: 0, mt: 0, ec: 0, ep: 0, fuelC: 0 });
  const [outputData, setOutputData] = useState<Array<number>>([]);

  const predictapi = async (inputData: InputTypes) => {
    const inputs = [inputData.m, inputData.mt, inputData.ec, inputData.ep, inputData.fuelC];
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: JSON.stringify({ inputs }),
      headers: { 'Content-Type': 'application/json' },
    });
    const output = await response.json();
    await setOutputData(output.prediction[0]);
  }

  return (
    <div className="Predict">
      <form>
        <input type="number" onChange={(e) => setInputData({...inputData, m: Number(e.target.value)})}/>
        <input type="number" onChange={(e) => setInputData({...inputData, mt: Number(e.target.value)})}/>
        <input type="number" onChange={(e) => setInputData({...inputData, ec: Number(e.target.value)})}/>
        <input type="number" onChange={(e) => setInputData({...inputData, ep: Number(e.target.value)})}/>
        <input type="number" onChange={(e) => setInputData({...inputData, fuelC: Number(e.target.value)})}/>
      </form>
      <button onClick={() => predictapi(inputData)}>Clique para prever</button>

      {outputData[0] && <div className="">Ewltp: {outputData[0]}</div>}
      {outputData[1] && <div className="">Enedc: {outputData[1]}</div>}

    </div>
  )
}

export default App
