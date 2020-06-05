import React, {Component} from 'react';

 

class Button extends Component {

 

    constructor(props) {
        super(props);
        

 

        this.rc = "right";
        if (props.orient === "left") {
            this.rc = "left" 
        }

 
        this.v = "not_active"
        if (props.visibility === true) {
            this.v = " ";
        }
        // console.log("first:",props);
}



    render() {
        let position = {
            top: this.props.top + '%',
            left: this.props.left + '%',
        };
        // console.log(position)
    let inc_dec;
    if (this.props.orient === "left") {
      inc_dec = this.props.minus
    }
    else {
      inc_dec = this.props.plus
    }
    

 

        return (
                <div className={this.v}>
                     <button onClick={inc_dec}  className="button" style={position} ><span className={this.rc}> &#x279C; </span></button>
                    {/*<a href="#"  className="button"  > &#x279C;</a>*/}
                </div>

            )
    }
}

 

export default Button;