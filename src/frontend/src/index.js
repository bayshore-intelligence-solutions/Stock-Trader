import React, { Component } from "react";
import ReactDOM from 'react-dom'; 
import Menu  from "./menu.js";
import Chart from "./chart.js";
import Table from "./table.js";
import Home from "./Home.js";
import "./styless.css";
import Button from './Button.js';
import './Button.css';
import {Router,Route} from 'react-router-dom';
import createBrowserHistory from 'history/createBrowserHistory';
import Dropdown from "./Dropdown.js";
// import {Switch} from 'react-router'; 

 

const history = createBrowserHistory();
const googleRoutes = [
    {
        id:0,
        routes:"GOOG/Table"
    },
    {
        id:1,
        routes:"GOOG/Graph"
    }
]

 

const appleRoutes = [
    {
        id:0,
        routes:"AAPL/Table"
    },
    {
        id:1,
        routes:"AAPL/Graph"
    }
]
class App extends Component {

    // update = () => {
    //     // let currentIndex
    //     this.state.currentIndex = -1
    //     console.log(this.state.currentIndex)
    // }

    constructor(props) {
        super(props);
        this.state = {
            currentIndex:-1
        }

 


}


handleRoutes = (param) => {
    var crrRts = window.location.pathname //  /AAPL
    console.log(crrRts.length)
    const{currentIndex} = this.state;
    let splitRoutes = crrRts.split('/');
    console.log(splitRoutes.length);
    console.log(splitRoutes);
    let checkOnce = crrRts.includes("GOOG") ? googleRoutes : appleRoutes
    console.log("---->",checkOnce);
    if(param === "left" && currentIndex !==0){
        this.setState({
            currentIndex : currentIndex-1 }, () => {
            let getRoutes = checkOnce[this.state.currentIndex];
            console.log("left:",getRoutes);

            history.push('/'+getRoutes.routes);        
        })
    }
    else if (param === "right" && currentIndex !==1){
         this.setState({
            currentIndex: currentIndex + 1 },() => {
            let getRoutes = checkOnce[this.state.currentIndex];
            history.push('/'+getRoutes.routes);
            console.log("right:",getRoutes);
        })
    } 
    else if (param === "left" && currentIndex ===0) {
        this.setState({
            currentIndex: currentIndex-1 },() => {
            history.push('/')
        })

    }      
}

 

 


render() {
    return ( 
          <Router history={history} >
            <div> 
            <Route path="/"   component={Menu} />
                <div className='sidenav'>
                   <Dropdown />
                </div>
                {/*<Dropdown />*/}
              {/*<Switch>    */}
                {/* For Apple*/}
              <div className='search'>
              <input
                type="text"

                ref="search"
                placeholder="type name here"
              />
              </div>

                <Route path="/" exact  component={Home} />
                {/*<Route path="/" exact component={Home} />*/}
                <Route path="/AAPL" exact 
                render={(props) => <div>
                                <Button plus={() => this.handleRoutes("right")} orient='right' visibility={true} top={90} left={60} />
                                <Button minus={() => this.handleRoutes("left")} orient='left' visibility={true} top={90} left={40} /> 
                                </div>}
                />
                <Route path="/AAPL/Table"  exact
                  render={(props) => <div> 
                                        <Table {...props} name="AAPL"/>
                                        <Button plus={() => this.handleRoutes("right")} orient='right' visibility={true} top={90} left={60} />
                                        <Button minus={() => this.handleRoutes("left")} orient='left' visibility={true} top={90} left={40} />  
                                        </div>}  
                />
                <Route path="/AAPL/Graph"  exact
                  render={(props) => <div> 
                                        <Chart {...props} name="AAPL"/>
                                        <Button plus={() => this.handleRoutes("right")} orient='right' visibility={true} top={90} left={60} />
                                        <Button minus={() => this.handleRoutes("left")} orient='left' visibility={true} top={90} left={40} />  
                                        </div>}  
                />
                {/* For Google */}
                <Route path="/GOOG" exact 
                    render={(props) =>  <div>
                                        <Button plus={() => this.handleRoutes("right")} orient='right' visibility={true} top={90} left={60} />
                                        <Button minus={() => this.handleRoutes("left")} orient='left' visibility={true} top={90} left={40} /> 
                                        </div>}
                />
                <Route path="/GOOG/Table"  exact
                  render={(props) => <div>
                                        <Table {...props} name="GOOG"/>
                                        <Button plus={() => this.handleRoutes("right")} orient='right' visibility={true} top={90} left={60} />
                                        <Button minus={() => this.handleRoutes("left")} orient='left' visibility={true} top={90} left={40} />
                                        </div>}
                />
                <Route path="/GOOG/Graph"  exact
                  render={(props) => <div>
                                        <Chart {...props} name="GOOG"/>
                                        <Button plus={() => this.handleRoutes("right")} orient='right' visibility={true} top={90} left={60} />
                                        <Button minus={() => this.handleRoutes("left")} orient='left' visibility={true} top={90} left={40} />
                                        </div>}
                />
              {/*</Switch>*/}
            </div>
{/*            <div>
                <Button plus={() => this.handleRoutes("right")} orient='right' visibility={true} top={85} left={60} />
                <Button minus={() => this.handleRoutes("left")} orient='left' visibility={true} top={85} left={40} />
            </div>*/}
          </Router>
    );
}
}

 

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);