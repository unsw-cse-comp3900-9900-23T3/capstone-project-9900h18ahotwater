function doSendEmail() {
    // check email format
    var email = document.getElementById("email").value;
    // if (!email.mach(/.+@.+\..+/)) {
    //     bootbox.alert({title:"Formate error",message:"Please enter a valid email address"});
    //     $("#email").focus();
    //     return false;
    // }

    // send email
    // parm = "email=" + email;
    // $.post("/reg", parm, function (data) {
    //     if (data == "success") {
    //         alert("success");
    //         // bootbox.alert({title:"Success",message:"We have sent you an email, please check it"});
    //     } else if (data == "timeout") {
    //         alert("timeout");
    //         // bootbox.alert({title:"Time out",message:"You have sent too many emails, please try again later"});
    //     } else {
    //         alert("error");
    //         // bootbox.alert({title:"Error",message:"Please try again"});
    //     }
    // });
    $.ajax({
        url: "/reg",
        type: "POST",
        contentType: 'application/json',
        data: JSON.stringify({'email': email}),
        success: function (data) {
           console.log(data)
            if (data['status'] == "success") {
                alert("success1")
                // bootbox.alert({title:"Success",message:"We have sent you an email, please check it"});
            } else if (data['status'] == "timeout") {
                alert("timeout1")
                // bootbox.alert({title:"Time out",message:"You have sent too many emails, please try again later"});
            } else {
                alert("error1")
                // bootbox.alert({title:"Error",message:"Please try again"});
            }
        },
        error: function () {
            // alert({title:"Error",message:"Please try again"});
            alert("error2")
        }
    });

}