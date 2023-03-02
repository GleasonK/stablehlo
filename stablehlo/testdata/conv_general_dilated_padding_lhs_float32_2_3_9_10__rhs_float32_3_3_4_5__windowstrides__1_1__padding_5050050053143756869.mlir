// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3x9x10xf32>, tensor<3x3x4x5xf32>)
    %1 = call @expected() : () -> tensor<2x3x9x9xf32>
    %2 = stablehlo.convolution(%0#0, %0#1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 2], [2, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2x3x9x10xf32>, tensor<3x3x4x5xf32>) -> tensor<2x3x9x9xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3x9x9xf32>, tensor<2x3x9x9xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3x9x10xf32>, tensor<3x3x4x5xf32>) {
    %0 = stablehlo.constant dense<"0x288848C050B28C3EB3D5543E6C5ABEBF962EACC09CE397403948B0BFB5BAE43F7DCC5CC0237FC0BE2E43784018F321BF49B14C406568D5C066E2374087339CBF69864E3F0CD394C00AAB58405E4126C01725E8BE7791A640AE8606C0CFE6CBBF75A4843FDDB24C40D1ACD93F84F514BF7258C5C09F0F41405264E7BF5D0199BFC4697DBF6ACD5F40595768406E56D53F6E11C53E2CE3AABF7F0D3E3FDC0A0D40179642BFB474DE409A07564008DA8D40803F49C05D98F0BF5FFA93408D4FB73FEA097DC0A9F39CC0AC7F2D40BD4EE140A166E5BDA6E7933F6AD99A3EB6C6D8BFAF7756C04EBBA840D2C8553F87C761C08E953940583FA6BF7931A43E843B8CC0A0F7DC3F36F8FA40B8300EC0A1AFA540F4EA17400F15BFC0A1D48FBF02F22B3E1ABB4940360156C0D3D78CC0BB6CF33E8C5703BCEBD5E5C0641C82C0BA0617C079AD8D408DF9CC3E561251C05CC450C00FEBB7C003D954BF7DE7E340FD4EB0C0464F1CC07C4DDA3F3D7A06406710763F18E26BC0AC7034BFC71E63401292C8BDB03A0D408B7FB03E3CCA2D401959943FD371A1BE78FC05C0F8CEEFBE42C32BBB899C0840C0AAC03F53DAD0BFBB54693FEFC134BEA399E6BFF2645D407C0B2DBDF8004FC09522FBBF637A34C01CAAC63D9CA4B5BF1A593F40453302BFD95854BF67C0EDBF848F203F2DCC2340179DB9BF4F1B733FF8C184C04E80ED3DF491174037FB0740798A03C0CD6BDEBFFC2D1AC0732B1B40DB1BE6BF4F35434093FD06BF160C0140F057DD3ED9C690C08972993F8F445CBF2F393ABC1159C63E31D9D03E5DA058C02237AD3FC8560C3F0128E0C030AB68BFF737C8BEA7A104C05C9C2A3F9262973EDF691C3F14BB2940F64BC9BF467D9B3FB99A11C02B15CFBFFD860F4047C704BEDA7893C050407BC02D206BBF057323C0094A233F984BF6BF76ECA0C0939F343F6E0D75C0E79D2F40CA62754091A83AC0E2043DC01B6276C0B7E35A40F23EEF3DE4F2573F8CE8953FDA27643F2072B8401C8F89407E2E2140D621EEBC7A491B40158C48BFB8C6A9BE0A4ACBBF015DA13F03C6813DE8F65A3F3D7F03BF8528983F5BFB3DBF1A439C3E4CD81AC0E80CAF40792FACC00D5FDB3FF9B8C13F10B9023FDC29823E08F58EC0C4F811C0BD07A8BF5EC87E404E7BA53F0E0E0EBF7E00363F1E2AA3BE7B777FC0B9A185C0F79B87BEDDE26840275730C07513A74012E58E3FF2767ABEC2E73DC070D08440EFD894C0204946C09C70B5BE21ECD53EA9050940E9B9303F2C3C86407B5D48BF38C09D3FA56774BF45575E3F4514FFBFB5F806C04EDB6E40318D97BFD46428403DEC83BF67D438405CA874BEC0F4C93F006DC1BF25B7A540D1C4E63F4566F33F0190C13D7AB6D1BEF6421340756A89C069A6D33F5C1C903FE22E3140B4D76E3DD8F8B8C089965940E8CC12BF2572CDBFFB72A4BD81DB58BE814131C0B432F4BF45816FC0A3D70840F317833FDA4ED33E177B563ECA973640ADBCA14026B696BF2B56513D7F4A93C06D1BA5BFDE03533FFBDBB7BDAE061BC046DDCC3F1125BBBFB5DCD6BF554A4140838CC7C038CA1AC047EDC5C0E3B38840F4D2C03F38D70B40FD76ABBF51D3914016089DC0FAB2233D79F6B6BF3BCD28BD123994BEC23287407BBF88C04AD51C4042870F4027D1953FAF42ED3E1967964003CE7D40D2B554BE675243BEFC03C5BE71160040B2F4F33F435DB93E16BECE3E5E9B8C3FE5147E40B465DEBF37CB0E40D6BA1F4021E48D3F20B9A4BEC975AC3D3C79303F44CE8140946F9240E9078D3FD7583040F413D7408C9A433FA3274B3F743C00C05F3D3D405250F6BFA5946CBF2CB6FEBF1DA29AC04851F3BF0A635FBE12EB6CC07B139D401B14D5BDBD9EACBE8911353FDEF1D33F337EC73F1B62EE3F043E5A40D9D3D4C03139633F3329A03DF1B4A2400346273FD199FFBE5F3AEC3E506444C0014AFB3FC5FBE7BF343C84C0EE0CDFBF2CAAA33F8F96C83FFBF00540E0F9D640CF5BDD3F82498CC0633030408A6697C0D1CF09C17CD91B406CBBCCBF914285C070088EC02973383F4C66E1BDCDFF2DC08FE059BFDE94CDBF954923C0A0469FBE65B76C3E30B58A3FD9775BC0B8A2AE3F2DCEB03FAF651AC021142E3FB6B0E13E6E64E93E384A8ABE855B2C40C29C983FE97B9DC0F9D086C07BF0A6BFFBDB9B40C95D6540A7F98E3E4C700BC0BD5F5BC0F56803C0651FBFBF394BADC081E7463F547549C09560A6BE1E1F2640C07CB7C0E44665C091B20A404865973E07639F3EEA33ECC0A2C3813FD7E98740807513BF1ED32040229CF43F97D415C009E5CC3F27FCCD3F7CB664BFCA6EB3BD1DFDF1BF83D3A3C088C31B40D8F13ABFFEEAB33E49831D4091CCB4C0710C5140D3347CC045798240467C7AC06FDD2D40506DB1409B3DF03E6976CEC0EC749E408537BBC0C9B884C054EF533E6A90E33F4F23963D066962C0370CB6BEEC5AF9BF44839F402AEF7D4098FDA7C042D565BF9C876F3F579E26BF7ED35EC0DBD4B63FDC0EEEBE48D579BFB26036C06D521AC0AC818D40900D133E181B98C0937D6ABF0B0864C0F6C62EBD5E3D89409ECDEABE38596940AA948A406A5093404DC690C04AABC7C02EA852C0BBCDCFBF47786F40A3878E40DFA3DEBD20F682C0963A8D3FC79A0240C0448FBF184F88BF454C97C0939583C0EE43823F2BB89CC0B063F43EAA74D2BFAF250D405DDA5E3CF0B62EC00FAD0A3F5A4F6F40EFA3F2BFC1C403C0BE236EC0AF50F83FF3C084C0745CBFBFF22325C09074463FD7739BBFDA5D8DBF7C5C6A401B0BC03ED319A3C0A0F0B0BE450D0F407498E5400B5497C04108EF3F98446CC020ADE0C0DB15903F1E0B1240E1A9EC3FEE35F03F16EF6DBFD31704C0B0014D3FF33D753FB2E3B7BF448E4240F4FB80C046C3593E438CB53FBF674FC0390CC83F9A0006C02DB40140A50E44BF23534BC03274AFBE2873C03F25B225BF27DBF2BFBB19613FB7F488BE7983A4BEC6CE2ABFE19117C0942E0240A27F77409693A2BF8BA26EC0ECE459C07E5ED13F79975BC01B34BD3F"> : tensor<2x3x9x10xf32>
    %1 = stablehlo.constant dense<"0x216179C00F588F3D7850AE3F77399FBF14A31AC03AEF6C4040C4F8BE30E152C061E6A040EF3C7BC0702195C0D6A5A13FC04C0740254146BEA0D83DC0789A6C40DBEC9FBF4AC47CC0381080C06B3E1F40C78FC43FA963DB3F0B9287C0663BC0BFAE13BEBFDD981440C7D1973F4040EC4055790BC01A2900C043207BC0325035BD662BEABFC4AB2140D4E1D1C051F2B33E67DDBB3F3A4092403F621BBDD3E78940C67DB53D070E53C0906539C08FEC69BF92188E40625E04C09B20CABF54B5ADC09E47D83F3DD194BF4E4B4A3FDEF189C065FE15407771A940E6D18240EA53B7BF38781940DFFA55C08169B63D37D5BF3F9161C6BC6F6EB8BF218922C03C1F91404B8E234003E24FC0417E14C00E3D8EC08EEE863D1B93AB3E3F884DBF8E407D40C84F1DBF6812A5C02A5FCDBFC89A4BC0EF45BB408A427140978A983F709622C0F1695CC03352C43FE5E5AF40B95C8C3FDDC04EC095B685BFC6929DC09B3B11C0C921D0406829833FA33820408F4AA8BE73AFA3BF5B65423F47C4494067BE31C039E6B8BF61276E3FF6F5F0BFA9A657409EB1CCBF07341F409105EFBFA2778B3F414A6F3FB4C529BFD87BCF3F04A19D40CB613740BD2E8FBF1666D13F27B47BBF412928406B8AA0BFA96A30C0237FE93E1F158EBF3A615B3FC791614081343D3E075732BFCC6484406DA60EC09BCD55BEF81356405DCB843F38F9F5BF9877933E45C1E7BFC451CEBD89302640A1E8C53FDBEB4C3F330B363C9FF9B13F78AFD63F09AB6840B5D8AB401E5810C0AEC1EBBF08E82640127B5CC085090A402877733F18C12E40AA589FC0949B9DC05215CA3F3D3A753F7B83883F918FBBBFA65BBFC09A95ACBFF3561F3E7F34C54078B099BFC97ECDBF15780E3ECAF53A3F4DF1F2BF9B5309C00BBF703F4ED80F40D4021EBF3CB0493F8387D9C094D9F53FE78F643F742E25BFB552E83FFEFAC33F79FD6FC0EFDF3540027C13BF7ADAA2BE7302E3BF0D239FC06717B13DF2C54840605C37BD"> : tensor<3x3x4x5xf32>
    return %0, %1 : tensor<2x3x9x10xf32>, tensor<3x3x4x5xf32>
  }
  func.func private @expected() -> tensor<2x3x9x9xf32> {
    %0 = stablehlo.constant dense<"0x5D9DC5C1C0B38DBEAFE4B9C2707864C2776576419E330EC2FAB8D6C11624284223149C428A42BEC152438441930337C2EA7F82C20CB43C412394B24130476C41F0C36842A7F813C2610306428FBAE2C1E202DBC1A03D3342887C8642F15FC84121F00EC3A06E1141A2019A42B3E80EC33E0D4B428390AB42BA17EB41CD540A418BC7A4C286524942D226F2C242E328413CB10042805657C194FCE5427C070041804502BF04B7AD4182C0C3C2104236422C3352C27B650C424C21F5400887A8C1C073FDC0224905C25A17B6C2E76E2642060BA8C2045C544205006CC2A07885BF8E919241488EC64108137642538A67C280F21E3F54B51243123591C2E9648C422C47094208315DC2F6ADEBC27BA427420811BC4248F15ABFF4E891C244BF31C1D4EB2E41607D9442BE79A84243D02BC224F3C3C2632EBD4212D17FC23BBB904136420E420039344292415441F0BD2D42418FF941E0C0434106179242107C8242A93589C2D4CECDC04D7F21C20CA4ACC0A276CCC1F6459AC196380D427A6978428AFB44C149363541CEA283C13875ACBFF087D6C2729FBDC16D8BD8C11093F6414E312342004C37BE86A5884213DFC5C2D82E7540E662D9C164B988418CD08EC00A459341FECDCCC2647CCD41986A1742CB41C742E9965DC26C84F141E087D73F9B2F35C2548FB141F08D06C1DF4953C26B067041CE37FB42165467C07CDB23C2D41E2D41A6751CC2F524B5C22892A2423D132D410E70F1C220AA0A401A25024282C71342C4A56F405EE24442B4DEE5C216E71C41708700433483C6C29451EFC28DC837C280C2B4C11E43D94038EECC4269EB4E42FD638EC2EA31F9417A51BE42805097C2F36C26C14BCC85C2FB5C96C20FCEB3C19D480A436A448F42F7334AC2393009C1268AEEC0473A2441BA9B714188D59D409450C640E18CE8418819EAC07836CD40EA300BC22891E4C19E3FF7C144E751C2209A1E404975A642EE46614263527B41301C8F419AAF58C1F2D60EC162E446C2AD9DD0412B800F42DAC89A424C70F7C04C4B6542A63EB24241558D412CF3D2C2AEBE09C14DB4FE41548FF64245214842D2B1A641E48C8FC1881E0DC230858340866BA641495FB8416283FE41AE52F841C43780404071FDC10E70FD41D2557CC11D0E87418720024224B59AC1FFF657C18D716D424B0A2B42901812426AE62FC2B362F841627864C2D6F11642FEE1C441881D5942440A4842784D33C04ACFA4C231E198C2FA7DCCC18138B2C234ACF4BFB6C882C24C9B7140006D58C2BE89B3422206824242D684C1AE244542E50D7CC2E015AD420A380B40FC7531C27FAF37C24C67B4414457AB41BC31F3C150DA61C0106837C28CFA11C2649A93C25CA165C26A58D8C164D0A2C19B851E42D447EC413BC00C42B6D80AC3C046D5BE56FE26C2B8D5FBC0D2E452C1AFF1D6C2292FE342FFA8E9C1DC5B9EC29E622C4208C1A2C1FD0C2AC2342ECFC1DD4B1E43E16BB242C674BFC23A209BC221B7AF42D531054223BD494128A835C2121706429E2635C2DE891FC270DB874102A2E6C2EC685641B8638B42962B7D42A07496C0DBD99741F7BE75C29EE783414ED348411A7AB6418007AFBF0BC5C9415423584294E15C42E8C966C2B7F2FB4272E42FC2084DA0C228445FC2A56D3D4250FBABC20085214145B63FC22DC08FC20F9294C1D48CE442A41B8CC2201CFE41808AADC1A8176542D22C37C28ED694C296B470C0CE2551428C838CC2658309423E9A9EC2F016FB4090B38F40765B51424B940CC220B12AC14A81A7C1C602A340B1F91EC2A098A83FF079C64263D4874153AE5FC24AEE29428E4AE6C21FC90FC27B582AC2BA168D411050554194FC914255B76A419AA58D4241534242263514C2620D8EC2CA42DCC26D039942501C554023B097C1547430C294F0084052D95D427E9B2EC252F9B3C24EE1AEC22E392342B41F174242CAE1C1BFE89FC282D490C1C7359FC1AE0F2BC14E0C24C23060D03FD420A2C14D6AB1C27646DF422BA29FC2FA2B24C3EE4D98C25D126442664B8CC2057A2CC26FD3EF421B11D6C17654DFC1FA03D5418A476DC240E1D7C1CE479EC1401EA8BE3ADD4CC2A495D0C1D372C54274D6CD41AF81A04208C32343C28F8342D57034C2CFEC5A42F7FF3CC147FB90C1F42A1E4284162EC22F90AEC23FB73A4209E5E0409DD7A9C10C46E4C1FA614FC210441BC1020F7241860F9B4290991342A7ABB5C1B2A4F6C18A0E1E41548EC24111EFB9407C6715C29C3F9CC24A203CC2B21586C22C660C42D914CD40ECAC06C2B6989DC25E55F6417C53E5422AC158423A41A841B439A342010709C032EA6FC1F6A657C2CAB094C2683DB441A661A442EE98EE4206A93D41BE51A6416E9A7BC2B0FE184204E37BC1F60425C2F0B7173F89FC1D4236E2AF422EAD9C42741D5241A272BBC1C04F2DC0485DE9C1F7D3D3C1881B7B4273681243E8F194422869EE4069280AC2361842C23C133FC27C8F07C1B4F4EEC1FA4F7CC24A7A684223018042CDCD21C2AFBE10C2525E0B42C6D039C2BA5459C1A7EBFE4158C6BA42C51189420A95CB4241F505C29B5153429C56C441A91736C2C60DE64228063F42C0045EC296F21E4242523CC1145DFF4269473C4120394CC2AC488BC1CF825842B869B741CF3A84423A0FD4C048257942690B5FC0346919C1EE068B411EBF9EC07B834942527ECEC01C82B93F42D9FCC18E0130C13D2D3C415624FF4012E28542"> : tensor<2x3x9x9xf32>
    return %0 : tensor<2x3x9x9xf32>
  }
}
