import React, { Component } from "react";
// import ReactDOM from 'react-dom'; 
import BootstrapTable from "react-bootstrap-table-next";
import "bootstrap/dist/css/bootstrap.min.css";
import filterFactory, { dateFilter} from 'react-bootstrap-table2-filter';
import paginationFactory from 'react-bootstrap-table2-paginator';


class Table extends Component {
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
        filter: dateFilter()
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
	    <div className="dataframe">
			
		    	<BootstrapTable keyField="Date"  
		    					data={this.state.posts} 
		    					columns={columns} 
		    					filter={ filterFactory() }
		    					filterPosition="bottom"
		    					pagination={ paginationFactory() }
		    	/>
		    
		</div>
    );
  }
}

export default Table;