import { useQuery } from 'react-query'
import axios from "axios"

function App() {
  
  const { data } = useQuery("predict", () => {
    return axios.get(`https://localhost:5000/predict/`).then((response)=> response.data)
  })

  return (
    <div className="Predict">
      
    </div>
  )
}

export default App
