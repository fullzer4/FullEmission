import { useState } from 'react';
import "../scss/Predict.scss"


function Predict() {

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

  const handleSubmit = (e:any) => {
    e.preventDefault();
    if (inputData.m && inputData.mt && inputData.ec && inputData.ep && inputData.fuelC) {
      predictapi(inputData);
    } else {
      alert('Por favor, preencha todos os campos antes de continuar.');
    }
  }

  return (
    <div className="Predict">
      <div className="text">

        <h1>AI for Car CO2 Emission</h1>
        <p>Neural Newtwork</p>

      </div>

      <form className='inputs'>

        <input type="number" onChange={(e) => {
          if(e.target.value) setInputData({...inputData, m: Number(e.target.value)}); 
          else setInputData({...inputData, m: 0});
        }}/>
        <input type="number" onChange={(e) => {
          if(e.target.value) setInputData({...inputData, mt: Number(e.target.value)}); 
          else setInputData({...inputData, mt: 0});
        }}/>
        <input type="number" onChange={(e) => {
          if(e.target.value) setInputData({...inputData, ec: Number(e.target.value)}); 
          else setInputData({...inputData, ec: 0});
        }}/>
        <input type="number" onChange={(e) => {
          if(e.target.value) setInputData({...inputData, ep: Number(e.target.value)}); 
          else setInputData({...inputData, ep: 0});
        }}/>
        <input type="number" onChange={(e) => {
          if(e.target.value) setInputData({...inputData, fuelC: Number(e.target.value)}); 
          else setInputData({...inputData, fuelC: 0});
        }}/>

        <input type="button" onClick={handleSubmit}></input>

      </form>

      <div className="output">

        <h2>Result</h2>

        {outputData[0] && <p className="">Ewltp: {outputData[0]}</p>}
        {outputData[1] && <p className="">Enedc: {outputData[1]}</p>}

      </div>
    </div>
  )
}

export default Predict
