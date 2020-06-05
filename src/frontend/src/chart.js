import React, { Component } from "react";
// import ReactDOM from 'react-dom'; 
// import BootstrapTable from "react-bootstrap-table-next";
// import "bootstrap/dist/css/bootstrap.min.css";
// import filterFactory, { dateFilter} from 'react-bootstrap-table2-filter';
// import paginationFactory from 'react-bootstrap-table2-paginator';
import { ComposedChart, Area, Line, XAxis, YAxis, Tooltip, Brush} from 'recharts';
import moment from 'moment';



// const dateFormatter = (item) => moment(item).format("DD MMM YY")
class Chart extends Component {
  constructor(props){
    super(props);

    this.state={
      posts: []
    }
  }

  // showSettings (event) {
  //   event.preventDefault();
  // }

  componentDidMount() {
    const url = "http://127.0.0.1:5000/" + this.props.name;
    fetch(url, {
      method: "GET"
    }).then(response => response.json()).then(posts => {this.setState({posts: posts})
    })

  }
  
  render() {

    const columns= [
      {
        dataField: "Date",
        text: "Date",
        // filter: dateFilter()
      },
      {
        dataField: "Open",
        text: "Open",
      },
      {
        dataField: "High",
        text: "High",
      },
      {
        dataField: "Low",
        text: "Low",
      },
      {
        dataField: "Close",
        text: "Close",
      },
      {
        dataField: "Adj Close",
        text: "Adj Close",
      },
      {
        dataField: "Volume",
        text: "Volume",
      }
    ]
    return ( 
    	<div>
    		<div className="title"> 
    		<h1>{this.props.name}</h1>
    		</div>

			<div className="chart">
	          <ComposedChart width={1400} height={480} 
	          data={this.state.posts} columns={columns} 
	          margin={{ top: 5, right: 20, bottom: 20, left: 0 }}>
            <Brush 
            dataKey='Date' 
            height={40} 
            stroke="#000000" 
            y={430} 
            startIndex={0}
            endIndex={10}>

            <ComposedChart>
              <Line type="monotone" dataKey="Date" stroke="#03308B" 
                dot={false} strokeWidth={2} />
            </ComposedChart>

            </Brush>
	          <defs>
	            <linearGradient id="colorUv" x1="0" y1="0" x2="0" y2="1">
	              <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
	              <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
	            </linearGradient>
	          </defs>
	          <Tooltip />
	          <XAxis dataKey="Date"  
	            tickFormatter={timeStr => moment(timeStr).format('DD MMM [\r\n] YY')}
	            fontSize={12} 
	            interval={19}
	            
	           />
	          <Line type="monotone" dataKey="Close" stroke="#03308B" 
	          dot={false} strokeWidth={2} activeDot={{ r: 8 }}/>
	          <Area type="monotone" dataKey="Open" stroke="#03308B" 
	          fillOpacity={0.3} fill="url(#colorUv)" activeDot={{ r: 0 }}/>
	          <YAxis/>




	          </ComposedChart>

        	</div>
		</div>
    );
  }
}

export default Chart;

