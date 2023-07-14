function [Dic_all, train_all, test_all] = read_data()
    
    load('YaleB.mat')
    data=descr;
    
    param.dictionary_size = 32; 
    param.train_size = 16;
    param.test_size = 16;

    [ Dic_all(1), train_all(1), test_all(1) ] = split_data( data,label,param );
    [ Dic_all(2), train_all(2), test_all(2) ] = split_data( data,label,param );
    [ Dic_all(3), train_all(3), test_all(3) ] = split_data( data,label,param );
    [ Dic_all(4), train_all(4), test_all(4) ] = split_data( data,label,param );
    [ Dic_all(5), train_all(5), test_all(5) ] = split_data( data,label,param );

    image = zeros(32*8, 32*16);
    for i = 1:8
        for j = 1:16
            starti = 32*(i-1);
            startj = 32*(j-1);
            image(starti + 1:starti+32,startj + 1:startj+32) = reshape(data(:, (i-1)*38 + j), 32, 32);
        end
    end
    figure, imshow(image, [])

end

