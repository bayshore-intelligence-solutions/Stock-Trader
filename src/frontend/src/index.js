import React, { Component } from "react";
import ReactDOM from 'react-dom'; 
import Menu  from "./menu.js"
import Table from "./table.js"
import "./styless.css";
import {BrowserRouter as Router,Route} from 'react-router-dom';

class App extends Component {  

  // triggerTable(sym){
  //   <Table name = {sym}/>
  // }
  render() {
    // var tableData = this.triggerTable("AAPL")
    return (  
      <Router>
        <div>
          <Menu title='Stock Trader' />
          {/*<Switch>*/}
            {/* For Apple*/}
            <Route path="/AAPL"  exact
              render={(props) => <Table {...props} name="AAPL"/>}  
            />
            {/* For Google */}
            <Route path="/GOOG"  
              render={(props) => <Table {...props} name="GOOG"/>}
            />
          {/*</Switch>*/}
        </div>
      </Router>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);





