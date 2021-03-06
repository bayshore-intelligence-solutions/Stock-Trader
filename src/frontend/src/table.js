import React, { Component } from "react";
import BootstrapTable from "react-bootstrap-table-next";
import "bootstrap/dist/css/bootstrap.min.css";
// import paginationFactory from 'react-bootstrap-table2-paginator';
// import filterFactory, { dateFilter} from 'react-bootstrap-table2-filter';
import 'react-bootstrap-table2-toolkit/dist/react-bootstrap-table2-toolkit.min.css';
import ToolkitProvider, { CSVExport } from 'react-bootstrap-table2-toolkit';


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
    const { ExportCSVButton } = CSVExport;
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

          <ToolkitProvider
            keyField="Date"
            data={ this.state.posts }
            columns={ columns }
              exportCSV={ {
                        ignoreHeader: true,
                        noAutoBOM: false
                      } }
          >
            {
              props => (
                <div className='dataframe'>

                <div>
                  <BootstrapTable { ...props.baseProps } />
                </div>
                <div className='expo'>
                  <ExportCSVButton { ...props.csvProps }>Export CSV!!</ExportCSVButton>
                  
                </div>
                </div>
              )
            }
          </ToolkitProvider>

      </div>
    );
  }
}

export default Table;
