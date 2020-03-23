import React, { Component } from "react";
import {Link} from 'react-router-dom';



// import { Dropdown, DropdownToggle, DropdownMenu, DropdownItem } from 'reactstrap'



class MenuLinks extends Component {

  myfun(){
      var displayOption = document.querySelector('.sidenav').style.display;
      var option;
      if (displayOption === 'inline-block') {
        option = 'none'
      }else {
        option = 'inline-block'
      }
      document.querySelector('.sidenav').style.display = option;
      //   var dropdown = document.getElementsByClassName("sidenav");
      //   var i;

      // for (i = 0; i < dropdown.length; i++) {
      //   dropdown[i].addEventListener("click", function() {
      //   this.classList.toggle("active");
      //   var dropdownContent = this.nextElementSibling;
      //   if (dropdownContent.style.display === "block") {
      //   dropdownContent.style.display = "none";
      //   } else {
      //   dropdownContent.style.display = "block";
      //   }
      //   });
      // }
  }
  constructor(props) {
    super(props);
    // Any number of links can be added here
    this.state = {
      buttons: [
        {
          text: 'Symbols',
          onClick: this.myfun,
          // icon: 'fa-pencil-square-o'
        }

      ]
    }
  }
  render() {
    let buttons = this.state.buttons.map(
      (button, i) => <li ref={i + 1}><button onClick={button.onClick}>{button.text}</button></li>);

    return (
        <div className={this.props.menuStatus} id='menu'>
          <ul>
            {buttons}
          </ul>

          <img className="logo" alt="logo" src={require('./logo.jpg')} />
           
          <ul className="sidenav">
            {/*<Link to="/AAPL">
              <li>AAPL</li>
            </Link>
            <Link to="/GOOG">
              <li>GOOG</li>
            </Link>*/}

             <li><Link onClick = {this.props.re} to="/AAPL">AAPL</Link></li>
             <li><Link onClick = {this.props.re} to="/GOOG">GOOG</Link></li>
          </ul>
        </div>

    )
  }
}

class Menu extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isOpen: false,
      symbol: "AAPL",
      viewForm: false
    }
    this._menuToggle = this._menuToggle.bind(this);
    this._handleDocumentClick = this._handleDocumentClick.bind(this);
  }
  componentDidMount() {
    document.addEventListener('click', this._handleDocumentClick, false);
  }
  componentWillUnmount() {
    document.removeEventListener('click', this._handleDocumentClick, false);
  }

 _handleDocumentClick(e) {
  if (!this.refs.root.contains(e.target) && this.state.isOpen === true) {
    this.setState({
    isOpen: false
  });
  };
}
_menuToggle(e) {
      console.log("part4:",this.state.isOpen);

  // var class_exist = document.getElementsByClassName('recharts-surface');
  // console.log("===============================>",class_exist.length);
  // if (class_exist.length>0)
  
  if (this.state.isOpen === false) {
    try{
      // document.querySelector('.recharts-surface').style.cssText = "width:800px !important; right:210px; position:absolute;transition: left 0.5s linear, margin-left 0.5s ease-out";
      console.log("part1"); 
      // document.querySelector('.recharts-wrapper').style.cssText = "width:800px !important;left:190px !important; position:absolute;transition: transform 500ms; ";
      // document.querySelector('.hambclicker').style.left="210px";

    }
    catch(err) {
      console.log('part3',err);
      document.querySelector('.hambclicker').style.left="210px";

    }

 
  }
  else{
    document.querySelector('.recharts-wrapper').style.cssText = "position:absolute;left:190px !important; transform: scaleX(1.10) scaleY(1.3) translateX(-15%);transition: transform 1s";
    document.querySelector('.hambclicker').style.left="8px";
    
    console.log("part2"); 
 
  }

  e.stopPropagation();
  this.setState({
    isOpen: !this.state.isOpen
  });
}


  render() {
    let menuStatus = this.state.isOpen ? 'isopen' : '';

    return (
      <div ref="root">
        <div className="menubar2">
          <div className="hambclicker" onClick={ this._menuToggle }></div>
          <div id="hambmenu" className={ menuStatus }><span></span><span></span><span></span><span></span></div>
{/*          <div className="title">
            <span>{ this.props.title }</span>
          </div>*/}
        </div>
        <MenuLinks re = {this.props.up} menuStatus={ menuStatus }/>
      </div>
    )
  }
}



export default Menu;