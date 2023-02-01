import { useState } from 'react';
import "../scss/Predict.scss"


function Predict() {

  interface InputTypes {
    m: number;
    mt: number;
    ec: number;
    ep: number;
    fuelC: number;
  }

  const [inputData, setInputData] = useState<InputTypes>({ m: 0, mt: 0, ec: 0, ep: 0, fuelC: 0 });
  const [outputData, setOutputData] = useState<Array<number>>([]);
  const [form, setForm] = useState("inputs");
  const [result, setResult] = useState("result");
  const [again, setAgain] = useState("again");

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

  const handleSubmit = (e: any) => {
    e.preventDefault();
    if (inputData.m && inputData.mt && inputData.ec && inputData.ep && inputData.fuelC) {
      setForm("inputs Ishow")
      setResult("result Rshow")
      setAgain("again Ashow")
      predictapi(inputData);
    } else {
      alert('Por favor, preencha todos os campos antes de continuar.');
    }
  }

  const reset = (e: any) => {
    e.preventDefault();
    setInputData({ m: 0, mt: 0, ec: 0, ep: 0, fuelC: 0 });
    setOutputData([]);
    setForm("inputs");
    setResult("result");
    setAgain("again");
    document.querySelectorAll('.input').forEach(input => {
      if (input instanceof HTMLInputElement) {
        input.value = '';
      }
    });
  }

  return (
    <div className="Predict">
      <div className="text">

        <h1>AI for Car CO2 Emission</h1>
        <p>Neural Network made with pytorch and python <br/>The algorithm return Ewltp and Enedc values </p>

      </div>

      <form className={form}>
        <div className='boxinputs'>
          <input className='input' type="number" placeholder='Weight (Kg)' onChange={(e) => {
            if (e.target.value) setInputData({ ...inputData, m: Number(e.target.value) });
            else setInputData({ ...inputData, m: 0 });
          }} />
          <input className='input' type="number" placeholder='Total weight (Kg)' onChange={(e) => {
            if (e.target.value) setInputData({ ...inputData, mt: Number(e.target.value) });
            else setInputData({ ...inputData, mt: 0 });
          }} />
        </div>
        <div className='boxinputs'>
          <input className='input' type="number" placeholder='Engine displacement (cm3) ' onChange={(e) => {
            if (e.target.value) setInputData({ ...inputData, ec: Number(e.target.value) });
            else setInputData({ ...inputData, ec: 0 });
          }} />
          <input className='input' type="number" placeholder="Engine power (KW)" onChange={(e) => {
            if (e.target.value) setInputData({ ...inputData, ep: Number(e.target.value) });
            else setInputData({ ...inputData, ep: 0 });
          }} />
        </div>
        <div className='boxinputs'>
          <input className='input' type="number" placeholder='Fuel consumption (L / 10Km)' onChange={(e) => {
            if (e.target.value) setInputData({ ...inputData, fuelC: Number(e.target.value) });
            else setInputData({ ...inputData, fuelC: 0 });
            }} />
        </div>

        <button onClick={handleSubmit}>Calculate</button>

      </form>

      <div className={result}>

          <div>

            <p>Results:</p>
            
            <div>
              <p className="">Ewltp: {outputData[0]}</p>
              <p className="">Enedc: {outputData[1]}</p>
            </div>

          </div>

      </div>

      <div className={again}>
            
            <button onClick={reset}>Reset</button>

      </div>
    </div>
  )
}

export default Predict
